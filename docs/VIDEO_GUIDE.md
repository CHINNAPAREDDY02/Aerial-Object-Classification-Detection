# Demo Video Guide

Use this as a practical script for the final project video. A 5-7 minute recording is enough.

## Recording Checklist

Before recording:

- Activate the virtual environment.
- Confirm dependencies are installed with `pip install -r requirements.txt`.
- Start the app with `streamlit run app/streamlit_app.py`.
- Keep one bird image, one drone image, and one detection test image ready.
- Open `results/reports/model_comparison.csv` or the Streamlit dashboard.

## Suggested Flow

### 1. Introduction, 30 seconds

Introduce the project:

> This project classifies aerial images as bird or drone and optionally detects objects with YOLOv8 bounding boxes. It includes trained classifiers, a detection model, preprocessing and training pipelines, evaluation reports, notebooks, and a Streamlit demo app.

### 2. Repository Tour, 60 seconds

Show these folders:

- `src/`: reusable model, preprocessing, evaluation, and prediction code.
- `scripts/`: training and command-line workflows.
- `notebooks/`: EDA, preprocessing, training, evaluation, and YOLO walkthroughs.
- `models/saved/`: trained classifier artifacts.
- `yolov8/`: detection configuration, training script, and YOLO weights.
- `results/`: plots and report outputs.
- `app/`: Streamlit user interface.

### 3. Dataset Format, 45 seconds

Show the classification dataset:

- `data/classification_dataset/TRAIN/bird`
- `data/classification_dataset/TRAIN/drone`
- `data/classification_dataset/VALID`
- `data/classification_dataset/TEST`

Show the detection dataset:

- `data/object_detection_dataset/images`
- `data/object_detection_dataset/labels`
- `yolov8/data.yaml`

Explain that YOLO labels use normalized bounding boxes:

```text
class_id center_x center_y width height
```

### 4. Training Pipelines, 60 seconds

Show:

- `scripts/train_all.py`
- `src/custom_cnn.py`
- `src/transfer_learning.py`
- `yolov8/train_yolo.py`

Mention:

- Custom CNN is trained from scratch.
- Transfer learning uses ResNet50, MobileNetV2, and EfficientNetB0.
- YOLOv8n is used for object detection.
- Metrics and plots are saved automatically under `results/`.

### 5. Streamlit App Demo, 90 seconds

Run:

```bash
streamlit run app/streamlit_app.py
```

Demonstrate:

- Single image classification.
- Confidence and bird/drone probability output.
- Batch classification and CSV export.
- YOLOv8 object detection on a sample image.
- Model performance dashboard.

### 6. Results and Comparison, 60 seconds

Show `results/reports/model_comparison.csv` or the dashboard.

Current saved comparison:

| Model | Accuracy | F1 | ROC AUC |
|---|---:|---:|---:|
| MobileNetV2 | 0.9767 | 0.9733 | 0.9990 |
| ResNet50 | 0.7953 | 0.7800 | 0.8887 |
| Custom CNN | 0.7814 | 0.7117 | 0.8529 |
| EfficientNetB0 | 0.6605 | 0.5466 | 0.6789 |

Say:

> Based on the saved evaluation summary, MobileNetV2 is the best current classifier.

### 7. Closing, 30 seconds

Close with:

> The final deliverables include trained models, Streamlit classification and detection, preprocessing/training/evaluation code, notebooks, comparison reports, documentation, and this demo video.

## Recommended Export

- Format: MP4
- Resolution: 1080p
- Filename: `aerial_object_classification_demo.mp4`
