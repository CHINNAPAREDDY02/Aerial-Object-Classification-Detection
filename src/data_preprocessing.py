"""
data_preprocessing.py — Data loading, augmentation, and generator pipeline
for the Aerial Bird vs Drone Classification project.
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    SHEAR_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP, VERTICAL_FLIP,
    BRIGHTNESS_RANGE, FILL_MODE, SEED
)


class _DirectoryBatchIterator:
    """Minimal directory iterator used when Keras cannot start its ThreadPool."""

    def __init__(self, directory, datagen, target_size, batch_size, shuffle, seed):
        self.directory = directory
        self.datagen = datagen
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.class_indices = {
            cls: idx
            for idx, cls in enumerate(
                sorted(
                    d for d in os.listdir(directory)
                    if os.path.isdir(os.path.join(directory, d))
                )
            )
        }
        self.num_classes = len(self.class_indices)
        self.filepaths = []
        self.classes = []

        for cls, idx in self.class_indices.items():
            cls_dir = os.path.join(directory, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.filepaths.append(os.path.join(cls_dir, fname))
                    self.classes.append(idx)

        self.samples = len(self.filepaths)
        self.filenames = [
            os.path.relpath(path, directory).replace("\\", "/")
            for path in self.filepaths
        ]
        self.classes = np.asarray(self.classes, dtype=np.float32)
        self._order = np.arange(self.samples)
        self._cursor = 0
        if self.shuffle:
            self.rng.shuffle(self._order)

    def __len__(self):
        return int(np.ceil(self.samples / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.samples == 0:
            raise StopIteration
        if self._cursor >= self.samples:
            self._cursor = 0
            if self.shuffle:
                self.rng.shuffle(self._order)

        batch_indices = self._order[self._cursor:self._cursor + self.batch_size]
        self._cursor += self.batch_size

        images = []
        labels = []
        for idx in batch_indices:
            img = Image.open(self.filepaths[idx]).convert("RGB").resize(self.target_size)
            arr = np.asarray(img, dtype=np.float32)
            arr = self.datagen.random_transform(arr)
            arr = self.datagen.standardize(arr)
            images.append(arr)
            labels.append(self.classes[idx])

        return np.stack(images, axis=0), np.asarray(labels, dtype=np.float32)


def _flow_from_directory(datagen, directory, target_size, batch_size, shuffle):
    """Use Keras' iterator, with a restricted-environment fallback."""
    try:
        return datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="binary",
            shuffle=shuffle,
            seed=SEED,
        )
    except PermissionError as exc:
        print(f"[data] Falling back to local iterator: {exc}")
        return _DirectoryBatchIterator(
            directory=directory,
            datagen=datagen,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=SEED,
        )


# ─────────────────────────────────────────────
# ImageDataGenerators
# ─────────────────────────────────────────────

def get_train_generator(train_dir: str = TRAIN_DIR,
                        batch_size: int = BATCH_SIZE,
                        img_size: tuple = IMG_SIZE):
    """
    Create an augmented ImageDataGenerator for the training set.

    Augmentations applied:
    - Normalization to [0, 1]
    - Random rotation, width/height shifts, shear, zoom
    - Horizontal flip
    - Brightness jitter
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        vertical_flip=VERTICAL_FLIP,
        brightness_range=BRIGHTNESS_RANGE,
        fill_mode=FILL_MODE,
    )

    generator = _flow_from_directory(datagen, train_dir, img_size, batch_size, shuffle=True)
    return generator


def get_valid_generator(valid_dir: str = VALID_DIR,
                        batch_size: int = BATCH_SIZE,
                        img_size: tuple = IMG_SIZE):
    """
    Create a non-augmented ImageDataGenerator for the validation set.
    Only normalizes pixel values.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = _flow_from_directory(datagen, valid_dir, img_size, batch_size, shuffle=False)
    return generator


def get_test_generator(test_dir: str = TEST_DIR,
                       batch_size: int = BATCH_SIZE,
                       img_size: tuple = IMG_SIZE):
    """
    Create a non-augmented ImageDataGenerator for the test set.
    shuffle=False to maintain label correspondence.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = _flow_from_directory(datagen, test_dir, img_size, batch_size, shuffle=False)
    return generator


def get_all_generators(batch_size: int = BATCH_SIZE, img_size: tuple = IMG_SIZE):
    """
    Convenience function — returns (train_gen, valid_gen, test_gen).
    """
    train_gen = get_train_generator(batch_size=batch_size, img_size=img_size)
    valid_gen = get_valid_generator(batch_size=batch_size, img_size=img_size)
    test_gen  = get_test_generator(batch_size=batch_size, img_size=img_size)
    print(f"\n[data] Class indices: {train_gen.class_indices}")
    print(f"[data] Train batches : {len(train_gen)}")
    print(f"[data] Valid batches : {len(valid_gen)}")
    print(f"[data] Test  batches : {len(test_gen)}")
    return train_gen, valid_gen, test_gen


# ─────────────────────────────────────────────
# tf.data pipeline (alternative to generators)
# ─────────────────────────────────────────────

def parse_image(file_path: tf.Tensor, label: tf.Tensor,
                img_size: tuple = IMG_SIZE, augment: bool = False):
    """
    Read and decode a JPEG/PNG image from disk, resize and normalize.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)

    return img, label


def build_tf_dataset(directory: str, img_size: tuple = IMG_SIZE,
                     batch_size: int = BATCH_SIZE, augment: bool = False,
                     shuffle: bool = True):
    """
    Build a tf.data.Dataset from a class-structured directory.

    Parameters
    ----------
    directory : str   — path like TRAIN_DIR with sub-folders per class
    augment   : bool  — apply augmentation (use only for training)
    """
    class_names = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ])
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    print(f"[data] Classes: {class_to_idx}")

    file_paths, labels = [], []
    for cls, idx in class_to_idx.items():
        cls_dir = os.path.join(directory, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(float(idx))

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=SEED)

    ds = ds.map(
        lambda x, y: parse_image(x, y, img_size, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, class_names


# ─────────────────────────────────────────────
# Visualize augmented samples
# ─────────────────────────────────────────────

def visualize_augmentation(image_path: str, n_augments: int = 8, save_path: str = None):
    """
    Show original image alongside n_augments augmented versions.
    """
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.7, 1.3),
        fill_mode="nearest",
    )

    img = load_img(image_path, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    x = x.reshape((1,) + x.shape)

    fig, axes = plt.subplots(1, n_augments + 1, figsize=(3 * (n_augments + 1), 3.5))
    axes[0].imshow(np.squeeze(x))
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    for i, batch in enumerate(datagen.flow(x, batch_size=1, seed=SEED)):
        if i >= n_augments:
            break
        axes[i + 1].imshow(np.squeeze(batch))
        axes[i + 1].set_title(f"Aug {i+1}", fontsize=9)
        axes[i + 1].axis("off")

    fig.suptitle("Data Augmentation Examples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# Dataset summary
# ─────────────────────────────────────────────

def dataset_summary():
    """Print a formatted dataset summary table."""
    from src.config import CLASSIFICATION_DIR

    print("\n" + "="*55)
    print("  AERIAL CLASSIFICATION DATASET SUMMARY")
    print("="*55)
    print(f"{'Split':<10} {'bird':>8} {'drone':>8} {'Total':>8}")
    print("-"*55)
    total_all = 0
    for split in ["TRAIN", "VALID", "TEST"]:
        split_dir = os.path.join(CLASSIFICATION_DIR, split)
        bird_n  = len(os.listdir(os.path.join(split_dir, "bird")))  if os.path.exists(os.path.join(split_dir, "bird"))  else 0
        drone_n = len(os.listdir(os.path.join(split_dir, "drone"))) if os.path.exists(os.path.join(split_dir, "drone")) else 0
        total = bird_n + drone_n
        total_all += total
        print(f"{split:<10} {bird_n:>8} {drone_n:>8} {total:>8}")
    print("-"*55)
    print(f"{'TOTAL':<10} {'':<8} {'':<8} {total_all:>8}")
    print("="*55 + "\n")


if __name__ == "__main__":
    dataset_summary()
    train_gen, valid_gen, test_gen = get_all_generators()
