import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Page configuration
st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")

st.title("AI Image Classifier (MobileNetV2)")
st.markdown("Upload an image to classify what it is using deep learning!")


# Cache the model so it doesn't reload on every action
@st.cache_resource
def load_model():
    with st.spinner("Loading AI model... Please wait."):
        model = MobileNetV2(weights="imagenet")
    return model


model = load_model()

# File uploader widget in Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image (fixed parameter here)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing image..."):
            # Preprocess the image for MobileNetV2
            img_array = np.array(image)
            img_resized = cv2.resize(img_array, (224, 224))
            img_preprocessed = preprocess_input(img_resized.astype(np.float32))
            img_batch = np.expand_dims(img_preprocessed, axis=0)

            # Predict
            preds = model.predict(img_batch)
            decoded_preds = decode_predictions(preds, top=3)[0]

            st.success("Analysis Complete!")
            st.subheader("Top Predictions:")

            for _, label, score in decoded_preds:
                st.write(f"- **{label.replace('_', ' ').title()}**: {score * 100:.2f}%")