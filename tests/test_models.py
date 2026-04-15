"""
tests/test_models.py — Unit tests for model architectures
Run: pytest tests/ -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_custom_cnn_build():
    """Custom CNN should build with correct input/output shapes."""
    try:
        import tensorflow as tf
    except ImportError:
        import pytest; pytest.skip("TensorFlow not installed")

    from src.custom_cnn import build_custom_cnn
    from src.config import INPUT_SHAPE

    model = build_custom_cnn()
    assert model is not None
    assert model.input_shape == (None,) + INPUT_SHAPE
    assert model.output_shape == (None, 1)  # Binary sigmoid
    assert model.count_params() > 0
    print(f"  Custom CNN params: {model.count_params():,}")


def test_custom_cnn_compile():
    """Custom CNN should compile without errors."""
    try:
        import tensorflow as tf
    except ImportError:
        import pytest; pytest.skip("TF not installed")

    from src.custom_cnn import build_custom_cnn, compile_model
    model = build_custom_cnn()
    model = compile_model(model)
    assert model.optimizer is not None
    assert model.loss is not None


def test_custom_cnn_forward_pass():
    """Custom CNN forward pass should produce values in [0, 1]."""
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        import pytest; pytest.skip("TF not installed")

    from src.custom_cnn import build_custom_cnn, compile_model
    from src.config import INPUT_SHAPE

    model = build_custom_cnn()
    model = compile_model(model)
    dummy = np.random.rand(2, *INPUT_SHAPE).astype("float32")
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (2, 1)
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def test_mobilenet_build():
    """MobileNetV2 should build successfully."""
    try:
        import tensorflow as tf
    except ImportError:
        import pytest; pytest.skip("TF not installed")

    from src.transfer_learning import build_mobilenet
    from src.config import INPUT_SHAPE

    model = build_mobilenet()
    assert model.input_shape == (None,) + INPUT_SHAPE
    assert model.output_shape == (None, 1)
    print(f"  MobileNetV2 params: {model.count_params():,}")


def test_resnet50_build():
    """ResNet50 should build successfully."""
    try:
        import tensorflow as tf
    except ImportError:
        import pytest; pytest.skip("TF not installed")

    from src.transfer_learning import build_resnet50
    model = build_resnet50()
    assert model.output_shape == (None, 1)
    print(f"  ResNet50 params: {model.count_params():,}")


def test_efficientnet_build():
    """EfficientNetB0 should build successfully."""
    try:
        import tensorflow as tf
    except ImportError:
        import pytest; pytest.skip("TF not installed")

    from src.transfer_learning import build_efficientnet
    model = build_efficientnet()
    assert model.output_shape == (None, 1)
    print(f"  EfficientNetB0 params: {model.count_params():,}")


def test_predict_single():
    """predict.py predict_from_pil should work with a dummy PIL image."""
    try:
        import tensorflow as tf
        from PIL import Image
        import numpy as np
    except ImportError:
        import pytest; pytest.skip("TF or PIL not installed")

    from src.custom_cnn import build_custom_cnn, compile_model
    from src.predict import predict_from_pil
    from src.config import IMG_SIZE, CLASSES

    model = build_custom_cnn()
    model = compile_model(model)

    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (*IMG_SIZE, 3), dtype=np.uint8)
    )
    result = predict_from_pil(dummy_img, model)

    assert "label" in result
    assert "confidence" in result
    assert result["label"] in CLASSES
    assert 0.0 <= result["confidence"] <= 100.0
