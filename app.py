import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "face_mask_model.h5"
TARGET_SIZE = (224, 224)

st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

page_style = """
<style>
/* Background grid */
.stApp {
    background-color: #080b14;
    background-image: 
        linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    color: #f8fafc;
}
/* Hide top header */
header {visibility: hidden;}
/* Remove padding */
.block-container {
    padding-top: 2rem;
    max-width: 800px;
}

/* File uploader custom styling */
[data-testid="stFileUploader"] {
    background-color: #111827;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 24px;
    margin-top: 1rem;
}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #0ea5e9 !important;
    border-radius: 12px !important;
    background-color: transparent !important;
    padding: 2rem !important;
}

/* The Browse Files button */
[data-testid="stFileUploader"] button {
    background-color: #38bdf8 !important;
    color: #0f172a !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
    margin-top: 1rem !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #0ea5e9 !important;
    color: #0f172a !important;
}
</style>
"""

st.markdown(page_style, unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; justify-content: center; margin-bottom: 2rem;">
    <div style="border: 1px solid #1e3a8a; border-radius: 4px; padding: 6px 16px; color: #38bdf8; font-family: monospace; font-size: 0.85rem; letter-spacing: 1px;">
        <span style="color: #38bdf8;">•</span> CNN FACE MASK DETECTOR
    </div>
</div>
<h1 style="text-align: center; font-family: monospace; font-size: 3.5rem; font-weight: 800; margin-bottom: 0; line-height: 1.1;">
    <span style="color: #ffffff;">Mask</span> <span style="color: #38bdf8;">Detection</span><br>
    <span style="color: #ffffff;">System</span>
</h1>
<p style="text-align: center; color: #94a3b8; font-family: sans-serif; margin-top: 1.5rem; font-size: 1.1rem; max-width: 500px; margin-left: auto; margin-right: auto; line-height: 1.5; margin-bottom: 2rem;">
    Upload a photo and the trained CNN model will predict whether<br>the person is wearing a face mask.
</p>
""", unsafe_allow_html=True)

model_status = st.empty()
model = None

if not os.path.exists(MODEL_PATH):
    model_status.markdown(
        f'<div style="background-color: #1e1b2e; border: 1px solid #4a3041; border-radius: 8px; padding: 12px 16px; color: #f87171; font-family: monospace; font-size: 0.9rem; margin-top: 1rem; margin-bottom: 1rem;">'
        f'X Could not load model — make sure {MODEL_PATH} is in the same directory'
        f'</div>', 
        unsafe_allow_html=True
    )
else:
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        model_status.markdown(
            f'<div style="background-color: #1e1b2e; border: 1px solid #4a3041; border-radius: 8px; padding: 12px 16px; color: #f87171; font-family: monospace; font-size: 0.9rem; margin-top: 1rem; margin-bottom: 1rem;">'
            f'X Failed to load model: {exc}'
            f'</div>', 
            unsafe_allow_html=True
        )

uploaded_file = st.file_uploader(
    "Drop an image or choose a file",
    type=["png", "jpg", "jpeg", "webp"],
    help="Upload a photo of a face for mask detection.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Unable to read the selected file as an image.")
        image = None

    if image is not None:
        st.image(image, caption="Input image", use_column_width=True)

        if model is None:
            st.warning("Model is not available. Please add the model file and reload.")
        else:
            if st.button("Analyse Image"):
                with st.spinner("Preprocessing image..."):
                    img = image.resize(TARGET_SIZE)
                    arr = np.array(img).astype("float32") / 255.0
                    arr = np.expand_dims(arr, axis=0)

                with st.spinner("Running inference..."):
                    prediction = model.predict(arr)
                    score = float(prediction[0][0])

                if score > 0.5:
                    label = "Without Mask"
                    badge_color = "#f87171"
                    confidence = int(score * 100)
                else:
                    label = "With Mask"
                    badge_color = "#34d399"
                    confidence = int((1 - score) * 100)

                st.markdown(
                    f"<div style='padding:16px; border-radius:16px; background:#111827; border:1px solid rgba(255,255,255,0.08);'>"
                    f"<p style='margin:0;font-size:0.9rem;color:#94a3b8;'>Prediction</p>"
                    f"<h2 style='margin:8px 0; color:{badge_color};'>{label}</h2>"
                    f"<p style='margin:4px 0 0; font-size:0.95rem;'>Confidence: <strong>{confidence}%</strong></p>"
                    f"<div style='margin-top:12px; width:100%; background:rgba(255,255,255,0.08); border-radius:999px;'>"
                    f"<div style='width:{confidence}%; height:10px; background:{badge_color}; border-radius:999px;'></div>"
                    f"</div>"
                    f"<p style='margin:12px 0 0; font-size:0.8rem;color:#94a3b8;'>Raw sigmoid output: <strong>{score:.6f}</strong></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem; margin-top: 3rem; color: #64748b; font-family: monospace; font-size: 0.85rem;">
    <div>Model: CNN · 3xConv2D · Sigmoid · 224x224 input</div>
    <div style="border: 1px solid rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 6px; cursor: pointer; transition: background 0.2s;" onmouseover="this.style.background='rgba(255,255,255,0.05)'" onmouseout="this.style.background='transparent'" onclick="window.location.reload()">
        ⟲ Reset
    </div>
</div>
""", unsafe_allow_html=True)
