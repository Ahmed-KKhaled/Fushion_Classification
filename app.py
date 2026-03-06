import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
import gdown


url = "https://drive.google.com/uc?id=1f6O4txTCVMGfeo7E1wGserXnoyfr8qQc"
output = "xception_v4_1_08_0.883.h5"

@st.cache_resource
def load_my_model():
    gdown.download(url, output, quiet=False, fuzzy=True)
    return load_model(output)

model = load_my_model()


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

# ========================
# Streamlit UI
# ========================
st.title("Clothing Classification App 👕")
st.write("Upload an image and the model will predict the clothing type.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((299, 299))  # Xception input size
    img = np.expand_dims(np.array(image), axis=0)
    img = preprocess_input(img)

    # Prediction
    pred = model.predict(img)
    class_id = np.argmax(pred)

    st.success(f"Prediction: {classes[class_id]}")