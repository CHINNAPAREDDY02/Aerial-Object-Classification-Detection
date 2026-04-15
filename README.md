# 🦅 Aerial Object Classification & Detection
### Bird vs Drone — Deep Learning Project

> Submission note: see `docs/DELIVERABLES.md` for the final deliverables checklist and `docs/VIDEO_GUIDE.md` for the demo recording plan.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)

---

## 📌 Problem Statement

A deep learning system that classifies aerial images as **Bird** or **Drone**
and optionally performs real-time object detection with bounding boxes.

**Applications:** Airport safety · Airspace surveillance · Wildlife research · Security

---

## 📁 Project Structure

```
aerial_project/
│
├── data/
│   ├── classification_dataset/
│   │   ├── TRAIN/  {bird/ drone/}
│   │   ├── VALID/  {bird/ drone/}
│   │   └── TEST/   {bird/ drone/}
│   └── object_detection_dataset/
│       ├── images/ {train/ val/ test/}
│       └── labels/ {train/ val/ test/}
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing_and_Augmentation.ipynb
│   ├── 03_Custom_CNN_Training.ipynb
│   ├── 04_Transfer_Learning.ipynb
│   ├── 05_Model_Evaluation_and_Comparison.ipynb
│   └── 06_YOLOv8_Object_Detection.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py               ← All hyperparameters & paths
│   ├── utils.py                ← Shared utilities & plotting
│   ├── data_preprocessing.py  ← Generators & augmentation
│   ├── custom_cnn.py           ← Custom CNN architecture
│   ├── transfer_learning.py   ← ResNet50 / MobileNet / EfficientNet
│   ├── evaluate.py             ← Metrics, confusion matrix, ROC, Grad-CAM
│   └── predict.py              ← Inference helpers
│
├── yolov8/
│   ├── data.yaml               ← YOLOv8 dataset config
│   └── train_yolo.py           ← Training, validation, inference, export
│
├── scripts/
│   ├── train_all.py            ← Master training pipeline
│   └── generate_sample_labels.py ← Auto-generate YOLO labels
│
├── app/
│   └── streamlit_app.py        ← Web application
│
├── models/
│   ├── saved/                  ← .h5 model files
│   └── checkpoints/            ← Training checkpoints
│
├── results/
│   ├── plots/                  ← Training curves, confusion matrices
│   └── reports/                ← JSON classification reports, CSV comparison
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / Extract
```bash
cd aerial_project
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option A — Run Everything at Once
```bash
# Train all 4 models, evaluate, compare, save best
python scripts/train_all.py

# Train specific models only
python scripts/train_all.py --models cnn,mobilenet --epochs 30
```

### Option B — Step-by-Step via Notebooks
```bash
jupyter notebook
```
Open notebooks in order: `01_EDA → 02 → 03 → 04 → 05 → 06`

### Option C — Train Individual Models
```bash
# Custom CNN
python -m src.custom_cnn

# Transfer Learning (one model)
python -m src.transfer_learning

# YOLOv8 Detection
python yolov8/train_yolo.py --mode train
python yolov8/train_yolo.py --mode val
python yolov8/train_yolo.py --mode infer --source data/object_detection_dataset/images/test
```

---

## 🖥️ Streamlit App

```bash
streamlit run app/streamlit_app.py
```

**Features:**
- 📸 Single image classification with confidence gauge
- 📦 Batch classification + CSV export
- 🎯 YOLOv8 detection with bounding boxes
- 📊 Model performance dashboard with comparison charts
- 🔧 Adjustable confidence threshold & IoU

---

## 📦 Packaged Artifacts

This repository includes the full implementation for:
- Custom CNN classification
- Transfer learning with ResNet50, MobileNetV2, and EfficientNetB0
- YOLOv8 object detection

The trained artifacts currently packaged inside the repository are:
- `models/saved/best_model.h5`
- `models/saved/custom_cnn.h5`
- `models/saved/resnet50_finetuned_phase1.h5`
- `yolov8/runs/yolov8n_bird_drone/weights/best.pt`

The MobileNetV2 and EfficientNetB0 training pipelines are implemented in code and notebooks, but their trained `.h5` exports are not bundled in this workspace snapshot.

---

## 🤖 Models

| Model | Type | Params | Speed | Notes |
|---|---|---|---|---|
| Custom CNN | From scratch | ~8M | Fast | Built with 4 Conv blocks |
| ResNet50 | Transfer | ~25M | Medium | Deep residual network |
| MobileNetV2 | Transfer | ~3.4M | Fastest | Best for deployment |
| EfficientNetB0 | Transfer | ~5.3M | Fast | Best accuracy/params ratio |
| YOLOv8n | Detection | ~3.2M | Real-time | Bounding box detection |

---

## 📊 Expected Performance

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Custom CNN | ~88% | ~87% | ~88% | ~87% |
| ResNet50 | ~93% | ~92% | ~93% | ~92% |
| MobileNetV2 | ~91% | ~90% | ~91% | ~90% |
| EfficientNetB0 | ~94% | ~93% | ~94% | ~93% |
| YOLOv8 | mAP@50: ~0.85 | — | — | — |

*Results vary depending on dataset size and hardware*

---

## 📈 Training Configuration

```python
# src/config.py — key settings
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-4
FINE_TUNE_LR  = 1e-5

# Augmentation
ROTATION_RANGE = 30
ZOOM_RANGE     = 0.2
HORIZONTAL_FLIP = True
```

---

## 🗂️ Dataset Structure

### Classification Dataset
```
TRAIN/  bird: 1414  drone: 1248
VALID/  bird:  217  drone:  225
TEST/   bird:  121  drone:   94
```

### Detection Dataset (YOLOv8 Format)
```
3319 images total | Train: 2662 | Val: 442 | Test: 215
Label format: <class_id> <cx> <cy> <width> <height>  (normalized)
Classes:  0 = bird  |  1 = drone
```

---

## 🛠️ Technical Stack

```
Deep Learning   : TensorFlow 2.x / Keras
Object Detection: YOLOv8 (Ultralytics)
Computer Vision : OpenCV, Pillow
ML Utilities    : scikit-learn, numpy, pandas
Visualization   : Matplotlib, Seaborn
Deployment      : Streamlit
Notebooks       : Jupyter
```

---

## 📤 Deliverables

- [x] Custom CNN (4-block architecture)
- [x] Transfer Learning pipeline (ResNet50, MobileNetV2, EfficientNetB0)
- [x] YOLOv8 Object Detection pipeline
- [x] Streamlit web application
- [x] 6 Jupyter notebooks (EDA → Training → Evaluation)
- [x] Model comparison report artifacts
- [x] Grad-CAM support in evaluation pipeline
- [x] Classification report artifacts
- [x] Master training script

---

## 🎓 Skills Demonstrated

`Deep Learning` · `CNN Architecture` · `Transfer Learning` ·
`Object Detection` · `Data Augmentation` · `Model Evaluation` ·
`YOLOv8` · `Streamlit` · `TensorFlow/Keras` · `Computer Vision` ·
`Grad-CAM` · `ROC/AUC Analysis` · `Confusion Matrix` · `F1 Score`

---

*Aerial Surveillance AI — Capstone Project*
