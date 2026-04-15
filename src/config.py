"""Shared paths, constants, and training settings for the project."""

import os


def _first_existing_path(*paths):
    """Pick the first file that already exists, with a sensible fallback."""
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]

# ─────────────────────────────────────────────
# Project Root
# ─────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# Dataset Paths
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT_DIR, "data")
CLASSIFICATION_DIR = os.path.join(DATA_DIR, "classification_dataset")
DETECTION_DIR = os.path.join(DATA_DIR, "object_detection_dataset")

TRAIN_DIR = os.path.join(CLASSIFICATION_DIR, "TRAIN")
VALID_DIR = os.path.join(CLASSIFICATION_DIR, "VALID")
TEST_DIR  = os.path.join(CLASSIFICATION_DIR, "TEST")

# ─────────────────────────────────────────────
# Model Save Paths
# ─────────────────────────────────────────────
MODELS_DIR       = os.path.join(ROOT_DIR, "models")
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, "saved")
CHECKPOINTS_DIR  = os.path.join(MODELS_DIR, "checkpoints")

CUSTOM_CNN_PATH       = os.path.join(SAVED_MODELS_DIR, "custom_cnn.h5")
RESNET50_PATH         = _first_existing_path(
    os.path.join(SAVED_MODELS_DIR, "resnet50_finetuned.h5"),
    os.path.join(SAVED_MODELS_DIR, "resnet50_finetuned_phase1.h5"),
)
MOBILENET_PATH        = os.path.join(SAVED_MODELS_DIR, "mobilenet_finetuned.h5")
EFFICIENTNET_PATH     = os.path.join(SAVED_MODELS_DIR, "efficientnetb0_finetuned.h5")
BEST_MODEL_PATH       = os.path.join(SAVED_MODELS_DIR, "best_model.h5")

# ─────────────────────────────────────────────
# Results Paths
# ─────────────────────────────────────────────
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# ─────────────────────────────────────────────
# Image Parameters
# ─────────────────────────────────────────────
IMG_SIZE      = (224, 224)   # Default image size used by the classifiers
IMG_CHANNELS  = 3
INPUT_SHAPE   = (224, 224, 3)
CLASSES       = ["bird", "drone"]
NUM_CLASSES   = 2

# ─────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-4
FINE_TUNE_LR  = 1e-5        # Smaller learning rate once the backbone is unfrozen
DROPOUT_RATE  = 0.5
L2_REG        = 1e-4
SEED          = 42

# ─────────────────────────────────────────────
# Data Augmentation Parameters
# ─────────────────────────────────────────────
ROTATION_RANGE        = 30
WIDTH_SHIFT_RANGE     = 0.2
HEIGHT_SHIFT_RANGE    = 0.2
SHEAR_RANGE           = 0.2
ZOOM_RANGE            = 0.2
HORIZONTAL_FLIP       = True
VERTICAL_FLIP         = False
BRIGHTNESS_RANGE      = (0.8, 1.2)
FILL_MODE             = "nearest"

# ─────────────────────────────────────────────
# Early Stopping & Callbacks
# ─────────────────────────────────────────────
EARLY_STOPPING_PATIENCE  = 10
REDUCE_LR_PATIENCE       = 5
REDUCE_LR_FACTOR         = 0.5
REDUCE_LR_MIN            = 1e-7
MONITOR_METRIC           = "val_accuracy"

# ─────────────────────────────────────────────
# YOLOv8 Configuration
# ─────────────────────────────────────────────
YOLO_MODEL_SIZE  = "yolov8n"   # Lightweight choice for faster training and inference
YOLO_EPOCHS      = 100
YOLO_IMG_SIZE    = 640
YOLO_BATCH       = 16
YOLO_PATIENCE    = 15
YOLO_CONF_THRES  = 0.25
YOLO_IOU_THRES   = 0.45
YOLO_YAML        = os.path.join(ROOT_DIR, "yolov8", "data.yaml")
YOLO_RUNS_DIR    = os.path.join(ROOT_DIR, "yolov8", "runs")

# ─────────────────────────────────────────────
# Class Mapping
# ─────────────────────────────────────────────
CLASS_NAMES = {0: "bird", 1: "drone"}
CLASS_COLORS = {
    "bird":  (0, 200, 100),   # Green
    "drone": (220, 50, 50),   # Red
}
