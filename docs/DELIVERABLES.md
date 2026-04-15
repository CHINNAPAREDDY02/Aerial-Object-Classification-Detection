# Project Deliverables

This file maps the requested project deliverables to the exact implementation artifacts in the repository.

## Trained Models

| Model | Type | Artifact |
|---|---|---|
| Custom CNN | Image classification | `models/saved/custom_cnn.h5` |
| ResNet50 | Transfer learning | `models/saved/resnet50_finetuned.h5` |
| MobileNetV2 | Transfer learning | `models/saved/mobilenet_finetuned.h5` |
| EfficientNetB0 | Transfer learning | `models/saved/efficientnetb0_finetuned.h5` |
| Best classifier | Selected model export | `models/saved/best_model.h5` |
| YOLOv8n | Object detection | `yolov8/runs/yolov8n_bird_drone/weights/best.pt` |

## Streamlit App

Run the app with:

```bash
streamlit run app/streamlit_app.py
```

Implemented app modes:

- Single image classification.
- Batch classification and CSV export.
- YOLOv8 object detection.
- Model performance dashboard.

## Scripts and Notebooks

Core modules:

- `src/config.py`: paths and hyperparameters.
- `src/data_preprocessing.py`: image generators and augmentation.
- `src/custom_cnn.py`: custom CNN architecture and training helpers.
- `src/transfer_learning.py`: ResNet50, MobileNetV2, and EfficientNetB0 models.
- `src/evaluate.py`: metrics, plots, reports, ROC curves, and confusion matrices.
- `src/predict.py`: inference helpers.
- `src/gradcam.py`: Grad-CAM utilities.
- `src/utils.py`: shared utilities.

Runnable scripts:

- `scripts/train_all.py`: full classifier training and comparison pipeline.
- `scripts/predict_cli.py`: command-line prediction.
- `scripts/demo.py`: local demo helper.
- `scripts/generate_sample_labels.py`: YOLO label generation helper.
- `yolov8/train_yolo.py`: YOLOv8 training, validation, inference, and export.

Notebooks:

- `notebooks/01_EDA.ipynb`
- `notebooks/02_Data_Preprocessing_and_Augmentation.ipynb`
- `notebooks/03_Custom_CNN_Training.ipynb`
- `notebooks/04_Transfer_Learning.ipynb`
- `notebooks/05_Model_Evaluation_and_Comparison.ipynb`
- `notebooks/06_YOLOv8_Object_Detection.ipynb`

## Model Comparison Report

Saved reports:

- `results/reports/model_comparison.csv`
- `results/reports/final_summary.json`
- `results/reports/custom_cnn_report.json`
- `results/reports/resnet50_report.json`
- `results/reports/mobilenetv2_report.json`
- `results/reports/efficientnetb0_report.json`

Saved plots:

- `results/plots/model_comparison.png`
- `results/plots/*_confusion.png`
- `results/plots/*_roc.png`
- `results/plots/*_history.png`

Current best model: `MobileNetV2`.

## GitHub Repository Documentation

Documentation files:

- `README.md`: setup, usage, repository structure, and summary results.
- `docs/DELIVERABLES.md`: deliverables-to-files checklist.
- `docs/VIDEO_GUIDE.md`: demo video script and recording checklist.
- `CONTRIBUTING.md`: contributor workflow.
- `Dockerfile`: containerized run option.
- `.github/workflows/ci.yml`: CI workflow.

## Code Structure

- `src/` contains reusable project logic.
- `scripts/` contains command-line workflows.
- `app/` contains the Streamlit interface.
- `yolov8/` contains object-detection configuration and training code.
- `tests/` contains unit tests for configuration, data, and model construction.

## Video

Use `docs/VIDEO_GUIDE.md` for a 5-7 minute recording plan covering the project goal, repository structure, dataset format, training artifacts, Streamlit classification, YOLOv8 detection, and model comparison results.
