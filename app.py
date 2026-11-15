

import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

MODEL_PATH = "best_pneumonia_model.keras"
FILE_ID = "1i0W4dcCneOB6z_MOmkxPInSa8DoJrjQv"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

st.title("🩺 Pneumonia Detection Using VGG16")

st.write(
    "Upload a chest X-ray image to get an AI-based prediction."
)

uploaded_file = st.file_uploader(
    "Choose a Chest X-Ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", width=350)

    if st.button("Predict"):
        try:
            with st.spinner("Analyzing X-ray image..."):

                model = load_model()

                resized_image = image.resize((224, 224))
                image_array = np.array(resized_image)
                image_array = np.expand_dims(image_array, axis=0)
                image_array = preprocess_input(image_array)

                prediction = model.predict(image_array)[0][0]

                if prediction >= 0.5:
                    st.error("Prediction: PNEUMONIA")
                    st.write(f"Pneumonia score: {prediction:.2%}")
                else:
                    st.success("Prediction: NORMAL")
                    st.write(f"Normal score: {(1-prediction):.2%}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.warning(
    "Educational prototype only. This model is not a "
    "clinically validated diagnostic tool. Consult a qualified doctor."
)
