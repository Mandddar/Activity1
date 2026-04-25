import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# Set page config for a premium feel
st.set_page_config(
    page_title="VisionAI - Transformer Classifier",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for premium aesthetics
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .header-container {
        padding: 2rem;
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
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
        background-image: linear-gradient(to right, #f093fb , #f5576c);
    }
    </style>
    """, unsafe_allow_html=True)

# Application Header
st.markdown('<div class="header-container"><h1 class="header-title">VisionAI Classifier</h1><p style="color: white; opacity: 0.8;">State-of-the-art Deep Learning with Transformers</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    """Load the Hugging Face image classification pipeline."""
    # Using a lightweight MobileNetV2 model from Hugging Face
    classifier = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
    return classifier

# Sidebar Content
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=100)
    st.title("Model Details")
    st.success("Using Vision Transformers / MobileNetV2")
    st.info("This version is optimized for cloud deployment with zero system dependencies.")

# Main Application Flow
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("🔍 Analysis Results")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing image using Deep Learning..."):
            classifier = load_classifier()
            
            # Run inference
            results = classifier(image)
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            for i, result in enumerate(results[:3]):
                label = result['label'].title()
                score = result['score']
                score_pct = score * 100
                
                st.write(f"**Rank {i+1}: {label}**")
                st.progress(float(score))
                st.write(f"Confidence: {score_pct:.2f}%")
                st.write("---")
            st.markdown('</div>', unsafe_allow_html=True)
            st.balloons()
    else:
        st.info("Upload an image to start the classification!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with Hugging Face & Streamlit</p>", unsafe_allow_html=True)
