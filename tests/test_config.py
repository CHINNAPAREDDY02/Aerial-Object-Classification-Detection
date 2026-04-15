"""
tests/test_config.py — Unit tests for src/config.py
Run: pytest tests/ -v
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_config_imports():
    """Config module must import without errors."""
    from src import config
    assert config is not None


def test_image_params():
    """Image parameters must be valid."""
    from src.config import IMG_SIZE, INPUT_SHAPE, NUM_CLASSES, CLASSES

    assert isinstance(IMG_SIZE, tuple)
    assert len(IMG_SIZE) == 2
    assert IMG_SIZE[0] > 0 and IMG_SIZE[1] > 0
    assert INPUT_SHAPE == (IMG_SIZE[0], IMG_SIZE[1], 3)
    assert NUM_CLASSES == 2
    assert "bird" in CLASSES
    assert "drone" in CLASSES


def test_hyperparams():
    """Hyperparameters must be within reasonable ranges."""
    from src.config import (
        BATCH_SIZE, EPOCHS, LEARNING_RATE, FINE_TUNE_LR,
        DROPOUT_RATE, L2_REG
    )
    assert 1 <= BATCH_SIZE <= 256
    assert 1 <= EPOCHS <= 1000
    assert 0 < LEARNING_RATE < 1
    assert 0 < FINE_TUNE_LR < LEARNING_RATE  # Fine-tune LR < initial LR
    assert 0.0 <= DROPOUT_RATE <= 1.0
    assert L2_REG >= 0


def test_augmentation_params():
    """Augmentation parameters must be valid."""
    from src.config import (
        ROTATION_RANGE, ZOOM_RANGE,
        HORIZONTAL_FLIP, BRIGHTNESS_RANGE
    )
    assert 0 <= ROTATION_RANGE <= 360
    assert 0 <= ZOOM_RANGE <= 1.0
    assert isinstance(HORIZONTAL_FLIP, bool)
    assert isinstance(BRIGHTNESS_RANGE, tuple)
    assert len(BRIGHTNESS_RANGE) == 2
    assert 0 < BRIGHTNESS_RANGE[0] <= BRIGHTNESS_RANGE[1]


def test_yolo_params():
    """YOLOv8 parameters must be valid."""
    from src.config import (
        YOLO_IMG_SIZE, YOLO_BATCH, YOLO_CONF_THRES, YOLO_IOU_THRES,
        CLASS_NAMES
    )
    assert YOLO_IMG_SIZE in (320, 416, 512, 640, 1280)
    assert YOLO_BATCH >= 1
    assert 0 < YOLO_CONF_THRES < 1
    assert 0 < YOLO_IOU_THRES < 1
    assert CLASS_NAMES[0] == "bird"
    assert CLASS_NAMES[1] == "drone"


def test_paths_exist_or_creatable():
    """Key directories should exist or be creatable."""
    from src.config import (
        ROOT_DIR, DATA_DIR, MODELS_DIR, RESULTS_DIR,
        PLOTS_DIR, REPORTS_DIR
    )
    # Root must exist
    assert os.path.isdir(ROOT_DIR), f"ROOT_DIR missing: {ROOT_DIR}"

    # Others can be created
    for path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, REPORTS_DIR]:
        os.makedirs(path, exist_ok=True)
        assert os.path.isdir(path), f"Could not create: {path}"


def test_class_colors():
    """Class color mappings must cover all classes."""
    from src.config import CLASS_COLORS, CLASSES
    for cls in CLASSES:
        assert cls in CLASS_COLORS, f"Missing color for class: {cls}"
        color = CLASS_COLORS[cls]
        assert len(color) == 3, "Color must be RGB tuple"
        for c in color:
            assert 0 <= c <= 255, f"Color channel out of range: {c}"
