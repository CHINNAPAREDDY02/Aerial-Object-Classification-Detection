"""
demo.py — Quick Project Demo Script
Runs through the entire pipeline end-to-end without requiring
pre-trained models. Uses a small subset of data for speed.

Run from project root:
    python scripts/demo.py
    python scripts/demo.py --quick   (3-epoch demo)
    python scripts/demo.py --full    (all models, full epochs)
"""

import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def banner(title: str, char: str = "═", width: int = 62):
    print(f"\n{'':>2}{char * width}")
    print(f"{'':>4}{title}")
    print(f"{'':>2}{char * width}")


def run_demo(quick: bool = True, full: bool = False):
    banner("🦅 AERIAL OBJECT CLASSIFICATION & DETECTION — DEMO")

    # ── Step 1: Verify environment ──────────────────────────
    banner("STEP 1 — Environment Check", "─")
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow  : {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"  ✅ GPUs        : {len(gpus)} available")
    except ImportError:
        print("  ❌ TensorFlow not installed. Run: pip install tensorflow")
        return False

    try:
        import ultralytics
        print(f"  ✅ Ultralytics : {ultralytics.__version__}")
    except ImportError:
        print("  ⚠️  Ultralytics not installed (YOLOv8 disabled)")

    try:
        import streamlit
        print(f"  ✅ Streamlit   : {streamlit.__version__}")
    except ImportError:
        print("  ⚠️  Streamlit not installed")

    # ── Step 2: Dataset summary ──────────────────────────────
    banner("STEP 2 — Dataset Summary", "─")
    from src.data_preprocessing import dataset_summary
    from src.utils import set_seed, ensure_dirs
    from src.config import PLOTS_DIR, REPORTS_DIR, SAVED_MODELS_DIR

    set_seed()
    ensure_dirs(PLOTS_DIR, REPORTS_DIR, SAVED_MODELS_DIR)
    dataset_summary()

    # ── Step 3: Data generators ──────────────────────────────
    banner("STEP 3 — Building Data Generators", "─")
    from src.data_preprocessing import get_all_generators
    train_gen, valid_gen, test_gen = get_all_generators(batch_size=16)
    print(f"  ✅ Class map   : {train_gen.class_indices}")
    print(f"  ✅ Train steps : {len(train_gen)}")
    print(f"  ✅ Valid steps : {len(valid_gen)}")
    print(f"  ✅ Test  steps : {len(test_gen)}")

    # Sample a batch
    images, labels = next(train_gen)
    print(f"  ✅ Batch shape : {images.shape}")
    print(f"  ✅ Pixel range : [{images.min():.3f}, {images.max():.3f}]")

    # ── Step 4: Model architecture ───────────────────────────
    banner("STEP 4 — Model Architecture Preview", "─")
    from src.custom_cnn import build_custom_cnn, compile_model
    from src.transfer_learning import build_mobilenet

    cnn = build_custom_cnn()
    print(f"  Custom CNN     params: {cnn.count_params():>12,}")

    mb = build_mobilenet()
    print(f"  MobileNetV2    params: {mb.count_params():>12,}")

    # ── Step 5: Quick training ───────────────────────────────
    n_epochs = 3 if quick else (50 if full else 10)

    banner(f"STEP 5 — Training Custom CNN ({n_epochs} epochs)", "─")
    from src.custom_cnn import train_custom_cnn

    t0 = time.time()
    model, history = train_custom_cnn(
        train_gen, valid_gen,
        epochs=n_epochs,
    )
    elapsed = time.time() - t0
    print(f"\n  ✅ Training complete in {elapsed:.0f}s")

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"  ✅ Best val accuracy: {best_val_acc:.4f}")

    # ── Step 6: Evaluation ───────────────────────────────────
    banner("STEP 6 — Model Evaluation", "─")
    from src.evaluate import evaluate_model
    test_gen.reset()
    results = evaluate_model(model, test_gen, model_name="Custom CNN (Demo)")

    print(f"\n  📊 Demo Results:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        val = results.get(metric, 0)
        bar = "█" * int(val * 20)
        print(f"  {metric.upper():<12}: {val:.4f}  {bar}")

    # ── Step 7: Single image prediction ─────────────────────
    banner("STEP 7 — Single Image Prediction", "─")
    from src.predict import predict_single
    from src.config import TRAIN_DIR

    for cls in ["bird", "drone"]:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.exists(cls_dir):
            files = os.listdir(cls_dir)
            if files:
                img_path = os.path.join(cls_dir, files[0])
                result = predict_single(img_path, model)
                status = "✅" if result["label"] == cls else "❌"
                print(f"  {status} [{cls.upper():>5}] → "
                      f"Predicted: {result['label'].upper():<6} "
                      f"({result['confidence']:.1f}% confidence)")

    # ── Step 8: Transfer learning preview (quick only) ───────
    if not quick:
        banner(f"STEP 8 — Transfer Learning Demo ({n_epochs} epochs)", "─")
        from src.transfer_learning import train_transfer_model
        ep1 = max(2, n_epochs // 3)
        ep2 = max(3, n_epochs // 2)
        mb_model, _, _ = train_transfer_model(
            "mobilenet", train_gen, valid_gen,
            epochs_phase1=ep1, epochs_phase2=ep2
        )
        test_gen.reset()
        mb_results = evaluate_model(mb_model, test_gen, model_name="MobileNetV2 (Demo)")
        print(f"  MobileNetV2 Accuracy: {mb_results['accuracy']:.4f}")

    # ── Step 9: YOLOv8 labels verification ───────────────────
    banner("STEP 8 — YOLOv8 Dataset Verification", "─")
    from yolov8.train_yolo import verify_dataset
    verify_dataset()

    # ── Summary ──────────────────────────────────────────────
    banner("🏁 DEMO COMPLETE", "═")
    print(f"  ✅ Pipeline verified successfully!")
    print(f"  ✅ Models    : {os.path.join(ROOT, 'models', 'saved')}")
    print(f"  ✅ Plots     : {os.path.join(ROOT, 'results', 'plots')}")
    print(f"  ✅ Reports   : {os.path.join(ROOT, 'results', 'reports')}")
    print(f"\n  To launch the full app:")
    print(f"    streamlit run app/streamlit_app.py")
    print(f"\n  To train all models:")
    print(f"    python scripts/train_all.py --models all --epochs 50")
    print(f"\n  To train YOLOv8:")
    print(f"    python yolov8/train_yolo.py --mode train")
    print()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial Project Demo")
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Quick 3-epoch demo (default)")
    parser.add_argument("--full",  action="store_true", default=False,
                        help="Full training demo (50 epochs)")
    args = parser.parse_args()

    success = run_demo(quick=not args.full, full=args.full)
    sys.exit(0 if success else 1)
