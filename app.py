import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Sugarcane Disease Detector", layout="centered")

st.title("🌿 Sugarcane Leaf Disease Detection App")
st.write("Upload a leaf image to detect: **Healthy, Rust, Red Rot, Blight**")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
 model = tf.keras.models.load_model("models/sugarcane_cnn.h5")
    return model

model = load_model()

# Class Labels
class_names = ["Healthy", "Rust", "Red Rot", "Blight"]

# ---------------------------
# Prediction Function
# ---------------------------
def predict(img):
    img = img.resize((224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = model.predict(img_arr)
    index = np.argmax(pred)
    confidence = np.max(pred) * 100

    return class_names[index], confidence


# ---------------------------
# Streamlit UI
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    if st.button("🔍 Diagnose Disease"):
        with st.spinner("Analyzing..."):
            result, score = predict(img)

        st.success(f"🌱 **Prediction:** {result}")
        st.info(f"📊 **Confidence:** {score:.2f}%")

        # Remedy suggestions
        remedies = {
            "Healthy": "No disease found. Keep monitoring the crop.",
            "Rust": "Apply Mancozeb or Propiconazole. Remove affected leaves.",
            "Red Rot": "Destroy diseased clumps. Use resistant varieties like Co 0238.",
            "Blight": "Spray Copper Oxychloride or Streptocycline. Improve field drainage."
        }

        st.warning(f"💡 Recommended Action: {remedies[result]}")

