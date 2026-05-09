import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Fashion Classification",
    page_icon="👕",
    layout="centered"
)

st.title("👕 Fashion Classification with Xception")
st.write("Upload a clothing image and the model will predict its category.")

# =========================
# Class Names
# =========================
class_names = [
    "Dress",
    "Hat",
    "Long Sleeve",
    "Outwear",
    "Pants",
    "Shirt",
    "Shoes",
    "Shorts",
    "Skirt",
    "T-shirt"
]

# =========================
# Model Config
# =========================
MODEL_PATH = "xception_v4_1_08_0.883.h5"

MODEL_URL = "https://huggingface.co/ahmed552005/fashion-classification-xception/blob/main/xception_v4_1_08_0.883.h5"

# =========================
# Download Model
# =========================
if not os.path.exists(MODEL_PATH):

    with st.spinner("Downloading model... Please wait ⏳"):

        response = requests.get(MODEL_URL)

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# Image Preprocessing
# =========================
IMG_SIZE = (299, 299)

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)

    img_array = tf.keras.applications.xception.preprocess_input(
        img_array
    )

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Prediction
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Predicting... 🔍"):

            processed_image = preprocess_image(image)

            predictions = model.predict(processed_image)

            predicted_index = np.argmax(predictions)

            confidence = np.max(predictions)

            predicted_class = class_names[predicted_index]

        st.success(f"Prediction: {predicted_class}")

        st.info(f"Confidence: {confidence * 100:.2f}%")

        # =========================
        # Probabilities
        # =========================
        st.subheader("Prediction Probabilities")

        probs = predictions[0]

        for i, class_name in enumerate(class_names):

            st.write(
                f"{class_name}: {probs[i] * 100:.2f}%"
            )

            st.progress(float(probs[i]))

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit & TensorFlow"
)
