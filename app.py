import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
import urllib.request
import os

# Set page config for a premium feel
st.set_page_config(
    page_title="VisionAI - Light DL Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium aesthetics
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .header-container {
        padding: 2rem;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: white;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #6a11cb , #2575fc);
    }
    </style>
    """, unsafe_allow_html=True)

# Application Header
st.markdown('<div class="header-container"><h1 class="header-title">VisionAI Classifier</h1><p style="color: white; opacity: 0.8;">Lightweight Deep Learning with Google MediaPipe</p></div>', unsafe_allow_html=True)

@st.cache_resource
def setup_classifier():
    """Download model and setup MediaPipe classifier."""
    model_path = 'model.tflite'
    if not os.path.exists(model_path):
        model_url = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
        urllib.request.urlretrieve(model_url, model_path)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
    classifier = vision.ImageClassifier.create_from_options(options)
    return classifier

# Sidebar Content
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/Google_MediaPipe_logo.png", width=100)
    st.title("Deployment Info")
    st.success("Successfully optimized for Streamlit Cloud!")
    st.info("Using EfficientNet Lite0 (TFLite) - high accuracy, low memory.")

# Main Application Flow
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("🔍 Analysis Results")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            classifier = setup_classifier()
            
            # Convert PIL image to MediaPipe Image format
            image_np = np.array(image.convert("RGB"))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            
            # Run inference
            classification_result = classifier.classify(mp_image)
            
            if classification_result.classifications:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                categories = classification_result.classifications[0].categories
                
                for i, category in enumerate(categories):
                    label = category.category_name.title()
                    score = category.score
                    score_pct = score * 100
                    
                    st.write(f"**Rank {i+1}: {label}**")
                    st.progress(float(score))
                    st.write(f"Confidence: {score_pct:.2f}%")
                    st.write("---")
                st.markdown('</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.error("No objects detected in the image.")
    else:
        st.info("Upload an image to see the deep learning model in action!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with MediaPipe & Streamlit</p>", unsafe_allow_html=True)
