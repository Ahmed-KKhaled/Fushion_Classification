import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# Load trained model
model = load_model("xception_v4_1_08_0.883.h5")

# Class names (modify according to your dataset)
classes = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "tshirt"
]

st.title("Clothing Classification App 👕")

st.write("Upload an image and the model will predict the clothing type.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((299,299))

    img = np.array(image)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    # Prediction
    pred = model.predict(img)

    class_id = np.argmax(pred)

    st.success(f"Prediction: {classes[class_id]}")