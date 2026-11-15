
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_pneumonia_model.keras")

st.title("🩺 Pneumonia Detection Using VGG16")
st.write("Upload a chest X-ray image for prediction.")

uploaded_file = st.file_uploader(
    "Choose a Chest X-Ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", width=350)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            model = load_model()

            image = image.resize((224, 224))
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)

            prediction = model.predict(image_array)[0][0]

            if prediction >= 0.5:
                st.error("Prediction: PNEUMONIA")
                st.write(f"Model score: {prediction:.2%}")
            else:
                st.success("Prediction: NORMAL")
                st.write(f"Model score: {(1-prediction):.2%}")

st.warning(
    "Educational prototype only. This model is not a "
    "clinically validated diagnostic tool. Consult a qualified doctor."
)
