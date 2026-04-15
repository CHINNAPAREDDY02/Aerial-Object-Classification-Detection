"""
predict_cli.py — Command-Line Inference Tool
Predict a single image or an entire folder from the terminal.

Usage:
    # Single image
    python scripts/predict_cli.py --image path/to/image.jpg

    # Entire folder
    python scripts/predict_cli.py --folder path/to/folder/

    # Choose model
    python scripts/predict_cli.py --image img.jpg --model models/saved/mobilenet_finetuned.h5

    # Change threshold
    python scripts/predict_cli.py --folder ./test_imgs --threshold 0.6

    # Save CSV results
    python scripts/predict_cli.py --folder ./test_imgs --save results.csv
"""

import os
import sys
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def predict_single_cli(args):
    """Predict a single image."""
    import tensorflow as tf
    from src.predict import load_model, predict_single
    from src.config import BEST_MODEL_PATH

    model_path = args.model or BEST_MODEL_PATH
    model = load_model(model_path)

    result = predict_single(args.image, model, threshold=args.threshold)

    print(f"\n{'─'*50}")
    print(f"  FILE       : {os.path.basename(args.image)}")
    print(f"  PREDICTION : {'🐦 BIRD' if result['label']=='bird' else '🚁 DRONE'}")
    print(f"  CONFIDENCE : {result['confidence']}%")
    print(f"  PROB BIRD  : {(1 - result['probability'])*100:.2f}%")
    print(f"  PROB DRONE : {result['probability']*100:.2f}%")
    print(f"{'─'*50}\n")


def predict_folder_cli(args):
    """Predict all images in a folder."""
    import tensorflow as tf
    from src.predict import load_model, predict_single
    from src.config import BEST_MODEL_PATH

    model_path = args.model or BEST_MODEL_PATH
    model = load_model(model_path)

    folder = Path(args.folder)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in folder.iterdir()
              if f.suffix.lower() in image_exts]

    if not images:
        print(f"[ERROR] No images found in: {folder}")
        return

    print(f"\n[predict] Processing {len(images)} images from: {folder}")
    print(f"{'─'*70}")
    print(f"{'File':<40} {'Label':<8} {'Confidence':>12} {'P(drone)':>10}")
    print(f"{'─'*70}")

    results = []
    bird_count = drone_count = 0

    for img_path in sorted(images):
        try:
            r = predict_single(str(img_path), model, threshold=args.threshold)
            label = r["label"]
            conf  = r["confidence"]
            prob  = r["probability"]

            if label == "bird":
                bird_count  += 1
                emoji = "🐦"
            else:
                drone_count += 1
                emoji = "🚁"

            fname = img_path.name[:38] + ".." if len(img_path.name) > 38 else img_path.name
            print(f"{fname:<40} {emoji} {label.upper():<6} {conf:>11.1f}% {prob*100:>9.2f}%")

            results.append({
                "filename":   img_path.name,
                "label":      label,
                "confidence": conf,
                "prob_drone": round(prob * 100, 2),
                "prob_bird":  round((1 - prob) * 100, 2),
            })
        except Exception as e:
            print(f"{img_path.name:<40} [ERROR] {e}")

    print(f"{'─'*70}")
    total = len(results)
    print(f"\n  Summary: {total} images | 🐦 Bird: {bird_count} | 🚁 Drone: {drone_count}")
    print(f"  Bird rate: {bird_count/total*100:.1f}%  | Drone rate: {drone_count/total*100:.1f}%")

    # Save CSV if requested
    if args.save:
        save_path = args.save
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  ✅ Results saved: {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="🦅 Aerial Object Classifier — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict_cli.py --image test.jpg
  python scripts/predict_cli.py --folder data/classification_dataset/TEST/bird
  python scripts/predict_cli.py --folder ./images --save predictions.csv
  python scripts/predict_cli.py --image img.jpg --model models/saved/custom_cnn.h5
        """
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",  type=str, help="Path to single image")
    mode.add_argument("--folder", type=str, help="Path to folder of images")

    parser.add_argument("--model",     type=str,   default=None,
                        help="Path to .h5 model (default: best_model.h5)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold 0-1 (default: 0.5)")
    parser.add_argument("--save",      type=str,   default=None,
                        help="Save results to CSV file")

    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] File not found: {args.image}")
            sys.exit(1)
        predict_single_cli(args)
    else:
        if not os.path.isdir(args.folder):
            print(f"[ERROR] Folder not found: {args.folder}")
            sys.exit(1)
        predict_folder_cli(args)


if __name__ == "__main__":
    main()
