import streamlit as st
import torch
# 🔥 FINAL FIX FOR PYTORCH 2.6+ (Streamlit Cloud / Python 3.13)
torch.serialization.set_default_weights_only(False
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# 🔥 CRITICAL FIX FOR TORCH 2.6+ (Python 3.13)
from ultralytics.nn.tasks import ClassificationModel
torch.serialization.add_safe_globals([ClassificationModel])

st.set_page_config(
    page_title="Plant Disease Diagnosis",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Plant Disease Diagnosis System")

# Load model (NO caching)
MODEL_PATH = "yolov8_plantvillage_model.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

model = YOLO(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Diagnose"):
        with st.spinner("Analyzing..."):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            results = model(temp_path)
            r = results[0]

            cls_id = int(r.probs.top1)
            confidence = float(r.probs.top1conf)
            label = model.names[cls_id]

            st.success(f"🦠 Disease: **{label}**")
            st.info(f"🔍 Confidence: **{confidence*100:.2f}%**")

            os.remove(temp_path)
