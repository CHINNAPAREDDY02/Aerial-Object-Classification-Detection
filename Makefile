# ============================================================
# Makefile — Aerial Object Classification & Detection Project
# ============================================================
# Usage:
#   make setup       → install all dependencies
#   make train       → train all models
#   make train-cnn   → train only Custom CNN
#   make train-tl    → train transfer learning models
#   make yolo-train  → train YOLOv8 detector
#   make yolo-val    → validate YOLOv8 detector
#   make app         → launch Streamlit app
#   make notebook    → launch Jupyter notebooks
#   make labels      → generate YOLOv8 labels from classified images
#   make clean       → remove generated model files & results
#   make help        → show this help

PYTHON      = python3
PIP         = pip
STREAMLIT   = streamlit
JUPYTER     = jupyter

.PHONY: all setup install train train-cnn train-tl yolo-train yolo-val \
        yolo-infer app notebook labels demo clean help

# ── Default ────────────────────────────────────────────────
all: setup

# ── Setup ─────────────────────────────────────────────────
setup: install labels
	@echo "✅ Project setup complete"

install:
	@echo "📦 Installing dependencies..."
	$(PIP) install -r requirements.txt

# ── Data ──────────────────────────────────────────────────
labels:
	@echo "🏷️  Generating YOLOv8 labels..."
	$(PYTHON) scripts/generate_sample_labels.py

# ── Training ──────────────────────────────────────────────
train:
	@echo "🚀 Training ALL models..."
	$(PYTHON) scripts/train_all.py --models all --epochs 50

train-fast:
	@echo "⚡ Fast training (20 epochs, CNN + MobileNet)..."
	$(PYTHON) scripts/train_all.py --models cnn,mobilenet --epochs 20

train-cnn:
	@echo "🧠 Training Custom CNN..."
	$(PYTHON) scripts/train_all.py --models cnn --epochs 50

train-tl:
	@echo "🔬 Training Transfer Learning models..."
	$(PYTHON) scripts/train_all.py --models resnet50,mobilenet,efficientnet --epochs 40

eval:
	@echo "📊 Evaluating saved models..."
	$(PYTHON) scripts/train_all.py --eval-only

# ── YOLOv8 ────────────────────────────────────────────────
yolo-train:
	@echo "🎯 Training YOLOv8..."
	$(PYTHON) yolov8/train_yolo.py --mode train

yolo-val:
	@echo "✅ Validating YOLOv8..."
	$(PYTHON) yolov8/train_yolo.py --mode val

yolo-infer:
	@echo "🔍 Running YOLOv8 inference on test set..."
	$(PYTHON) yolov8/train_yolo.py --mode infer \
		--source data/object_detection_dataset/images/test

yolo-export:
	@echo "📤 Exporting YOLOv8 to ONNX..."
	$(PYTHON) yolov8/train_yolo.py --mode export --format onnx

# ── App & Notebooks ───────────────────────────────────────
app:
	@echo "🌐 Launching Streamlit App..."
	$(STREAMLIT) run app/streamlit_app.py \
		--server.port 8501 \
		--server.headless false

app-server:
	@echo "🖥️  Launching Streamlit in headless mode (port 8501)..."
	$(STREAMLIT) run app/streamlit_app.py \
		--server.port 8501 \
		--server.headless true \
		--server.enableCORS false

notebook:
	@echo "📔 Launching Jupyter Notebook..."
	$(JUPYTER) notebook notebooks/

# ── Demo ──────────────────────────────────────────────────
demo:
	@echo "🎬 Running quick project demo..."
	$(PYTHON) scripts/demo.py

# ── Clean ─────────────────────────────────────────────────
clean-models:
	@echo "🗑️  Removing saved models..."
	rm -f models/saved/*.h5
	rm -f models/checkpoints/*

clean-results:
	@echo "🗑️  Removing results..."
	rm -f results/plots/*.png
	rm -f results/reports/*.json results/reports/*.csv

clean-yolo:
	@echo "🗑️  Removing YOLOv8 runs..."
	rm -rf yolov8/runs/

clean: clean-models clean-results clean-yolo
	@echo "✅ Cleaned all generated files"

clean-cache:
	@echo "🗑️  Removing Python cache..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ── Help ──────────────────────────────────────────────────
help:
	@echo ""
	@echo "  🦅 Aerial Object Classification & Detection"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup        Install deps + generate labels"
	@echo "  make train        Train ALL models (CNN + TL)"
	@echo "  make train-fast   Quick train: CNN + MobileNet (20 epochs)"
	@echo "  make train-cnn    Custom CNN only"
	@echo "  make train-tl     Transfer Learning models only"
	@echo "  make eval         Evaluate all saved models"
	@echo "  make yolo-train   Train YOLOv8 detector"
	@echo "  make yolo-val     Validate YOLOv8"
	@echo "  make yolo-infer   Run inference on test images"
	@echo "  make yolo-export  Export to ONNX"
	@echo "  make app          Launch Streamlit web app"
	@echo "  make notebook     Open Jupyter notebooks"
	@echo "  make demo         Quick demo script"
	@echo "  make clean        Remove all generated files"
	@echo "  make help         Show this message"
	@echo ""
