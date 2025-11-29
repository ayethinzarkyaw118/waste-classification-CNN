# app.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TensorFlow logs

import json, numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st
from keras.models import load_model  # Keras 3 API

IMG_SIZE = (200, 200)
RESCALE = 1 / 255.0

@st.cache_resource
def load_assets():
    # ---- Load model ----
    model_path = Path("waste_classifier.h5")
    if not model_path.exists():
        raise FileNotFoundError("waste_classifier.h5 not found next to app.py")
    model = load_model(model_path, compile=False)  # legacy H5 format

    # ---- Load labels ----
    labels_path = Path("labels.json")
    if not labels_path.exists():
        raise FileNotFoundError("labels.json not found next to app.py")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    # convert {name: index} -> {index: name}
    inv = {int(v): k for k, v in labels.items()}

    return model, inv

model, inv = load_assets()

def preprocess(img: Image.Image):
    x = np.array(img.convert("RGB").resize(IMG_SIZE), dtype="float32") * RESCALE
    return np.expand_dims(x, 0)  # shape: (1, H, W, 3)

st.title("♻ Waste Classifier")

up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if up:
    img = Image.open(up)
    st.image(img, caption="Input", use_container_width=True)

    p = model.predict(preprocess(img), verbose=0)

    if p.shape[1] == 1:
        prob1 = float(p[0, 0])
        scores = {0: 1.0 - prob1, 1: prob1}
    else:
        scores = {i: float(p[0, i]) for i in range(p.shape[1])}

    best = max(scores, key=scores.get)
    st.subheader(f"Prediction: {inv.get(best, best)} ({scores[best]:.2%})")

    st.caption("All class scores:")
    st.table([{ "class": inv.get(i, i), "prob": f"{scores[i]:.2%}" } for i in sorted(scores)])
else:
    st.info("Upload a photo to get a prediction.")

