"""Streamlit app for testing the bird-vs-drone models in one place."""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import streamlit as st

# ── Path setup ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    CLASSES, PLOTS_DIR, REPORTS_DIR,
    CUSTOM_CNN_PATH, RESNET50_PATH, MOBILENET_PATH,
    EFFICIENTNET_PATH, BEST_MODEL_PATH,
    YOLO_CONF_THRES, YOLO_IOU_THRES
)

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🦅 Aerial Object Classifier",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a73e8, #34a853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    /* Prediction card */
    .pred-card {
        padding: 20px 24px;
        border-radius: 14px;
        border: 2px solid;
        margin: 12px 0;
        text-align: center;
    }
    .pred-bird {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-color: #4CAF50;
        color: #2e7d32;
    }
    .pred-drone {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border-color: #f44336;
        color: #c62828;
    }
    .pred-label {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .pred-confidence {
        font-size: 1.2rem;
        font-weight: 600;
    }
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #1a73e8;
        margin: 6px 0;
    }
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #333;
        border-bottom: 2px solid #1a73e8;
        padding-bottom: 6px;
        margin-bottom: 10px;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #eee;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: Load model (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classification_model(model_path: str):
    """Load and cache a Keras classification model."""
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    """Load and cache a YOLOv8 detection model."""
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return model
    except Exception:
        return None


def resolve_yolo_weights_path() -> str:
    """Return the YOLO weights path that best matches the current workspace."""
    run_name = os.environ.get("YOLO_RUN", "yolov8n_bird_drone")
    candidates = [
        ROOT / "yolov8" / "runs" / run_name / "weights" / "best.pt",
        ROOT / "yolov8" / "runs" / "aerial_detection" / run_name / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def resolve_comparison_report_path() -> str:
    """Use the real comparison CSV when available, with a safe fallback."""
    candidates = [
        Path(REPORTS_DIR) / "model_comparison.csv",
        Path(REPORTS_DIR) / "model_comparison_template.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


# ─────────────────────────────────────────────
# Helper: Predict
# ─────────────────────────────────────────────
def classify_image(pil_image: Image.Image, model, threshold: float = 0.5) -> dict:
    """Run classification inference on a PIL image."""
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label_idx = int(prob >= threshold)
    label = CLASSES[label_idx]
    confidence = prob if label_idx == 1 else 1 - prob
    return {
        "label":     label,
        "confidence": round(confidence * 100, 2),
        "prob_drone": round(prob * 100, 2),
        "prob_bird":  round((1 - prob) * 100, 2),
        "raw":        prob,
    }


def detect_objects(pil_image: Image.Image, yolo_model,
                   conf: float, iou: float) -> tuple:
    """Run YOLOv8 detection on a PIL image. Returns annotated image + results."""
    import tempfile, cv2

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path)

    results = yolo_model.predict(
        source=tmp_path, conf=conf, iou=iou,
        imgsz=640, verbose=False
    )
    os.unlink(tmp_path)

    # Draw boxes on PIL image
    annotated = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    class_colors = {"bird": (76, 175, 80), "drone": (244, 67, 54)}
    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            xyxy     = box.xyxy[0].tolist()
            name     = yolo_model.names[cls_id]
            color    = class_colors.get(name, (33, 150, 243))

            draw.rectangle(xyxy, outline=color, width=3)
            label_text = f"{name.upper()} {conf_val:.0%}"
            draw.rectangle(
                [xyxy[0], xyxy[1] - 22, xyxy[0] + len(label_text) * 9, xyxy[1]],
                fill=color
            )
            draw.text((xyxy[0] + 3, xyxy[1] - 20), label_text, fill="white")
            detections.append({"class": name, "confidence": round(conf_val * 100, 1), "bbox": xyxy})

    return annotated, detections


# ─────────────────────────────────────────────
# Helper: Confidence gauge
# ─────────────────────────────────────────────
def confidence_gauge(confidence: float, label: str):
    """Render a horizontal confidence bar."""
    color = "#4CAF50" if label == "bird" else "#f44336"
    st.markdown(f"""
    <div style="margin:10px 0">
        <div style="display:flex;justify-content:space-between;font-size:0.9rem;color:#555;">
            <span>Confidence</span><span><b>{confidence:.1f}%</b></span>
        </div>
        <div style="background:#eee;border-radius:8px;height:18px;margin-top:4px;">
            <div style="width:{confidence}%;background:{color};border-radius:8px;
                        height:18px;transition:width 0.5s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)

    # App mode
    app_mode = st.selectbox(
        "🖥️ Application Mode",
        ["📸 Single Image Classification",
         "📦 Batch Classification",
         "🎯 Object Detection (YOLOv8)",
         "📊 Model Performance Dashboard"],
        index=0
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-header">🤖 Model Selection</div>', unsafe_allow_html=True)

    MODEL_PATHS = {
        "✨ Best Model (Auto)":  BEST_MODEL_PATH,
        "🧠 Custom CNN":         CUSTOM_CNN_PATH,
        "🔬 ResNet50":           RESNET50_PATH,
        "📱 MobileNetV2":        MOBILENET_PATH,
        "⚡ EfficientNetB0":     EFFICIENTNET_PATH,
    }

    chosen_model_name = st.selectbox("Classification Model", list(MODEL_PATHS.keys()))
    chosen_model_path = MODEL_PATHS[chosen_model_name]

    threshold = st.slider(
        "🎯 Decision Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Probability ≥ threshold → Drone; < threshold → Bird"
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-header">🔭 YOLOv8 Settings</div>', unsafe_allow_html=True)

    yolo_conf = st.slider("Confidence Threshold", 0.1, 0.9, YOLO_CONF_THRES, 0.05)
    yolo_iou  = st.slider("IoU (NMS) Threshold",  0.1, 0.9, YOLO_IOU_THRES,  0.05)

    yolo_weights_path = resolve_yolo_weights_path()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.82rem;color:#888;">
    🦅 Aerial Surveillance AI<br>
    Bird vs Drone Classification<br>
    Deep Learning | Computer Vision
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🦅 Aerial Object Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Deep Learning · Bird vs Drone Detection · Aerial Surveillance AI</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# Mode: Single Image Classification
# ─────────────────────────────────────────────
if "Single Image" in app_mode:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("📤 Image Source")
        source = st.radio("Choose source:", ["Upload File", "Select from Dataset"], horizontal=True, key="single_src")
        
        img = None
        if source == "Upload File":
            uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"], help="Upload an aerial image to classify as Bird or Drone")
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
        else:
            base_dir = os.path.join(ROOT, "data", "classification_dataset", "TEST")
            if os.path.exists(base_dir):
                d_col1, d_col2 = st.columns(2)
                cls_opt = d_col1.selectbox("Class Folder", ["bird", "drone"], key="single_cls")
                folder_path = os.path.join(base_dir, cls_opt)
                if os.path.exists(folder_path):
                    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if images:
                        selected_img = d_col2.selectbox("Select Image", images, key="single_img")
                        img_path = os.path.join(folder_path, selected_img)
                        img = Image.open(img_path).convert("RGB")
                    else:
                        st.warning("No images found in folder.")
            else:
                st.warning("Dataset TEST folder not found!")

        if img:
            st.image(img, caption="Selected Image", use_column_width=True)
            st.caption(f"Size: {img.size[0]} × {img.size[1]} px")

    with col_result:
        st.subheader("🔍 Prediction")

        if img:
            with st.spinner(f"Classifying with {chosen_model_name}..."):
                model = load_classification_model(chosen_model_path)

            if model is None:
                st.error(f"❌ Model not found at:\n`{chosen_model_path}`\n\nPlease train the model first.")
            else:
                result = classify_image(img, model, threshold=threshold)
                label  = result["label"]
                conf   = result["confidence"]

                # Prediction card
                css_class = "pred-bird" if label == "bird" else "pred-drone"
                emoji = "🐦" if label == "bird" else "🚁"
                st.markdown(f"""
                <div class="pred-card {css_class}">
                    <div class="pred-label">{emoji} {label.upper()}</div>
                    <div class="pred-confidence">Confidence: {conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                confidence_gauge(conf, label)

                # Probability breakdown
                st.markdown("**📊 Class Probabilities**")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric("🐦 Bird",  f"{result['prob_bird']:.1f}%")
                with prob_col2:
                    st.metric("🚁 Drone", f"{result['prob_drone']:.1f}%")

                # Probability bar chart
                fig, ax = plt.subplots(figsize=(5, 2.5))
                bars = ax.barh(
                    ["Bird 🐦", "Drone 🚁"],
                    [result["prob_bird"], result["prob_drone"]],
                    color=["#4CAF50", "#f44336"], height=0.45, edgecolor="white"
                )
                for bar, val in zip(bars, [result["prob_bird"], result["prob_drone"]]):
                    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                            f"{val:.1f}%", va="center", fontsize=10)
                ax.set_xlim(0, 115)
                ax.set_xlabel("Probability (%)")
                ax.set_title("Class Probability Distribution", fontsize=11)
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("👆 Upload an aerial image to get started.")
            st.markdown("""
            **What this model can detect:**
            - 🐦 **Birds** — various species in flight
            - 🚁 **Drones** — UAVs, quadcopters, fixed-wing

            **Use cases:**
            - Airport bird-strike prevention
            - Restricted airspace monitoring
            - Wildlife research & census
            - Security surveillance
            """)


# ─────────────────────────────────────────────
# Mode: Batch Classification
# ─────────────────────────────────────────────
elif "Batch" in app_mode:
    st.subheader("📦 Batch Image Classification")
    st.info("Process multiple images at once.")

    source = st.radio("Choose source for batch:", ["Upload Multiple Files", "Select Dataset Folder"], horizontal=True, key="batch_src")
    
    images_to_process = []
    
    if source == "Upload Multiple Files":
        uploaded_files = st.file_uploader("Choose multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            images_to_process = [(uf.name, Image.open(uf).convert("RGB")) for uf in uploaded_files]
    else:
        base_dir = os.path.join(ROOT, "data", "classification_dataset", "TEST")
        if os.path.exists(base_dir):
            cls_opt = st.selectbox("Select Test Folder to Process", ["bird", "drone"], key="batch_cls")
            folder_path = os.path.join(base_dir, cls_opt)
            if os.path.exists(folder_path):
                img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                st.write(f"Found {len(img_files)} images in `{cls_opt}` folder.")
                if st.button("Start Batch Processing") and img_files:
                    # Limit to 50 to prevent frontend lag
                    for f in img_files[:50]:
                        images_to_process.append((f, Image.open(os.path.join(folder_path, f)).convert("RGB")))
                    if len(img_files) > 50:
                        st.warning("Only processing the first 50 images to prevent lag.")
        else:
            st.warning("Dataset TEST folder not found!")

    if images_to_process:
        model = load_classification_model(chosen_model_path)
        if model is None:
            st.error(f"❌ Model not found: `{chosen_model_path}`")
        else:
            results_data = []
            cols = st.columns(min(4, len(images_to_process)))
            progress = st.progress(0)

            for i, (fname, img) in enumerate(images_to_process):
                result = classify_image(img, model, threshold)
                results_data.append({
                    "Filename":   fname,
                    "Prediction": result["label"].upper(),
                    "Confidence": f"{result['confidence']:.1f}%",
                    "Bird %":     f"{result['prob_bird']:.1f}%",
                    "Drone %":    f"{result['prob_drone']:.1f}%",
                })

                col = cols[i % len(cols)]
                with col:
                    st.image(img, use_column_width=True)
                    color = "🟢" if result["label"] == "bird" else "🔴"
                    st.caption(f"{color} **{result['label'].upper()}** ({result['confidence']:.0f}%)")

                progress.progress((i + 1) / len(images_to_process))

            # Summary table
            import pandas as pd
            df = pd.DataFrame(results_data)
            st.markdown("### 📋 Results Summary")
            st.dataframe(df, use_container_width=True)

            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv,
                               "aerial_predictions.csv", "text/csv")

            # Summary stats
            bird_count  = sum(1 for r in results_data if "BIRD" in r["Prediction"])
            drone_count = len(results_data) - bird_count
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Images", len(results_data))
            m2.metric("🐦 Birds",     bird_count)
            m3.metric("🚁 Drones",    drone_count)


# ─────────────────────────────────────────────
# Mode: YOLOv8 Object Detection
# ─────────────────────────────────────────────
elif "Detection" in app_mode:
    st.subheader("🎯 YOLOv8 Object Detection")
    st.info("Detect and localize birds/drones with bounding boxes.")

    source = st.radio("Choose image source:", ["Upload File", "Select from Detection Dataset"], horizontal=True, key="det_src")
    
    img = None
    if source == "Upload File":
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="det_up")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
    else:
        base_dir = os.path.join(ROOT, "data", "object_detection_dataset", "images", "test")
        if os.path.exists(base_dir):
            images = [f for f in os.listdir(base_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                selected_img = st.selectbox("Select Test Image", images, key="det_img")
                img_path = os.path.join(base_dir, selected_img)
                img = Image.open(img_path).convert("RGB")
            else:
                st.warning("No test images found.")
        else:
            st.warning("Dataset test folder not found!")

    if img:

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_column_width=True)

        yolo_model = load_yolo_model(yolo_weights_path)

        if yolo_model is None:
            st.warning(f"""
            ⚠️ YOLOv8 weights not found at:
            `{yolo_weights_path}`

            To train the detection model, run:
            ```bash
            python yolov8/train_yolo.py --mode train
            ```
            """)
        else:
            with st.spinner("Running YOLOv8 detection..."):
                annotated, detections = detect_objects(img, yolo_model, yolo_conf, yolo_iou)

            with col2:
                st.image(annotated, caption="Detection Results", use_column_width=True)

            if detections:
                st.success(f"✅ Found {len(detections)} object(s)")
                import pandas as pd
                det_df = pd.DataFrame(detections)
                st.dataframe(det_df, use_container_width=True)
            else:
                st.warning("No objects detected above confidence threshold.")
    else:
        st.info("👆 Upload an aerial image to run detection.")
        st.markdown("""
        **YOLOv8 Detection features:**
        - 📦 Real-time bounding boxes
        - 🏷️ Class labels with confidence scores
        - 🎛️ Adjustable confidence & IoU thresholds
        """)


# ─────────────────────────────────────────────
# Mode: Performance Dashboard
# ─────────────────────────────────────────────
elif "Dashboard" in app_mode:
    st.subheader("📊 Model Performance Dashboard")

    # Load comparison report if exists
    comparison_path = resolve_comparison_report_path()
    summary_path    = os.path.join(REPORTS_DIR, "final_summary.json")

    if os.path.exists(comparison_path):
        import pandas as pd
        df = pd.read_csv(comparison_path)

        st.markdown("#### 🏆 Model Comparison")
        st.dataframe(df.style.highlight_max(axis=0, subset=df.select_dtypes("number").columns,
                                             color="#d4edda"), use_container_width=True)

        # Radar / Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        metrics = ["accuracy", "precision", "recall", "f1"]
        metrics_exist = [m for m in metrics if m in df.columns]
        x = np.arange(len(df))
        width = 0.18
        colors = ["#3F51B5", "#4CAF50", "#FF9800", "#F44336"]

        for i, (metric, color) in enumerate(zip(metrics_exist, colors)):
            ax.bar(x + i*width, df[metric], width, label=metric.capitalize(),
                   color=color, alpha=0.85, edgecolor="white")

        ax.set_xticks(x + width * (len(metrics_exist)-1)/2)
        ax.set_xticklabels(df.get("Model", df.index), fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — Classification Metrics", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info("📭 No evaluation results found yet.\n\nTrain models first:\n```bash\npython scripts/train_all.py\n```")

    # Show saved plot images
    st.markdown("#### 📈 Training Curves & Confusion Matrices")
    plot_files = list(Path(PLOTS_DIR).glob("*.png")) if os.path.exists(PLOTS_DIR) else []

    if plot_files:
        grid_cols = st.columns(2)
        for i, plot_file in enumerate(sorted(plot_files)):
            with grid_cols[i % 2]:
                st.image(str(plot_file), caption=plot_file.stem.replace("_", " ").title(),
                         use_column_width=True)
    else:
        st.info("No plots found. Run training to generate visualizations.")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🦅 Aerial Object Classification & Detection System |
    Built with TensorFlow · YOLOv8 · Streamlit |
    Deep Learning Capstone Project
</div>
""", unsafe_allow_html=True)
