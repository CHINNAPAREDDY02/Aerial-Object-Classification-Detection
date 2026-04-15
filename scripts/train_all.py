"""
train_all.py — Master Training Pipeline
Trains Custom CNN + all Transfer Learning models, evaluates them,
compares results, and saves the best model.

Run from project root:
    python scripts/train_all.py
    python scripts/train_all.py --epochs 30 --batch 16 --models all
    python scripts/train_all.py --models cnn,mobilenet
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# ── Add project root to path ─────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    EPOCHS, BATCH_SIZE, PLOTS_DIR, REPORTS_DIR,
    CUSTOM_CNN_PATH, BEST_MODEL_PATH, SAVED_MODELS_DIR
)
from src.utils import set_seed, ensure_dirs, plot_model_comparison
from src.data_preprocessing import get_all_generators, dataset_summary
from src.custom_cnn import train_custom_cnn
from src.transfer_learning import train_transfer_model
from src.evaluate import evaluate_model, compare_models, select_best_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Aerial Classification Models")
    parser.add_argument("--epochs",  type=int, default=EPOCHS,
                        help=f"Training epochs (default: {EPOCHS})")
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--models",  type=str, default="all",
                        help="Comma-separated: all | cnn | resnet50 | mobilenet | efficientnet")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate saved models")
    return parser.parse_args()


def train_pipeline(args):
    """Full training + evaluation pipeline."""

    set_seed()
    ensure_dirs(PLOTS_DIR, REPORTS_DIR, SAVED_MODELS_DIR)

    # ── Dataset summary ───────────────────────
    dataset_summary()

    # ── Build data generators ─────────────────
    train_gen, valid_gen, test_gen = get_all_generators(
        batch_size=args.batch
    )

    # ── Select which models to train ──────────
    all_models = ["cnn", "resnet50", "mobilenet", "efficientnet"]
    if args.models == "all":
        selected = all_models
    else:
        selected = [m.strip() for m in args.models.split(",")]
        invalid = set(selected) - set(all_models)
        if invalid:
            print(f"[ERROR] Unknown models: {invalid}. Choose from {all_models}")
            sys.exit(1)

    model_objects = {}
    training_times = {}

    # ── Train ─────────────────────────────────
    if not args.eval_only:
        for model_name in selected:
            print(f"\n{'='*60}")
            print(f"  TRAINING: {model_name.upper()}")
            print(f"{'='*60}")
            t0 = time.time()

            if model_name == "cnn":
                model, _ = train_custom_cnn(
                    train_gen, valid_gen,
                    epochs=args.epochs,
                )
                model_objects["Custom CNN"] = model

            else:
                ep1 = max(10, args.epochs // 3)
                ep2 = max(15, args.epochs // 2)
                model, _, _ = train_transfer_model(
                    model_name, train_gen, valid_gen,
                    epochs_phase1=ep1,
                    epochs_phase2=ep2,
                )
                display = {
                    "resnet50":     "ResNet50",
                    "mobilenet":    "MobileNetV2",
                    "efficientnet": "EfficientNetB0",
                }[model_name]
                model_objects[display] = model

            training_times[model_name] = round(time.time() - t0, 1)
            print(f"[pipeline] {model_name} training time: {training_times[model_name]}s")

    else:
        # Load saved models for eval-only mode
        import tensorflow as tf
        from src.config import (CUSTOM_CNN_PATH, RESNET50_PATH,
                                MOBILENET_PATH, EFFICIENTNET_PATH)
        paths = {
            "Custom CNN":    CUSTOM_CNN_PATH,
            "ResNet50":      RESNET50_PATH,
            "MobileNetV2":   MOBILENET_PATH,
            "EfficientNetB0": EFFICIENTNET_PATH,
        }
        for name, path in paths.items():
            if os.path.exists(path):
                model_objects[name] = tf.keras.models.load_model(path)
                print(f"[pipeline] Loaded: {name} from {path}")

    # ── Evaluate all models on test set ───────
    print(f"\n{'='*60}")
    print("  EVALUATION ON TEST SET")
    print(f"{'='*60}")

    all_results = {}
    for name, model in model_objects.items():
        test_gen.reset()
        results = evaluate_model(model, test_gen, model_name=name)
        all_results[name] = results

    # ── Comparison ────────────────────────────
    if len(all_results) > 1:
        df = compare_models(all_results)

    # ── Select & save best ────────────────────
    if model_objects:
        best_name, best_model = select_best_model(
            all_results, model_objects, save_path=BEST_MODEL_PATH, metric="f1"
        )

    # ── Training time report ──────────────────
    if training_times:
        print("\n[pipeline] Training Duration Summary:")
        for name, t in training_times.items():
            minutes = t // 60
            seconds = t % 60
            print(f"  {name:<15}: {int(minutes)}m {int(seconds)}s")

    # ── Save final results JSON ───────────────
    summary = {
        "model_results": all_results,
        "training_times_sec": training_times,
        "best_model": best_name if model_objects else "N/A",
    }
    summary_path = os.path.join(REPORTS_DIR, "final_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\n[pipeline] ✅ All done! Summary saved: {summary_path}")

    return all_results


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args)
