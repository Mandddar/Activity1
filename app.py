import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config for a premium feel
st.set_page_config(
    page_title="VisionAI - Deep Learning Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium aesthetics
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Header styling */
    .header-container {
        padding: 2rem;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
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
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Upload area styling */
    .stFileUploader {
        border: 2px dashed #4facfe;
        border-radius: 15px;
        padding: 20px;
        background-color: rgba(79, 172, 254, 0.05);
    }
    
    /* Prediction card styling */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
    }
    
    /* Progress bar color */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe , #00f2fe);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Header
st.markdown('<div class="header-container"><h1 class="header-title">VisionAI Classifier</h1><p style="color: white; opacity: 0.8;">Harnessing MobileNetV2 for Real-time Image Intelligence</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained MobileNetV2 model."""
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image):
    """Resize and prepare the image for the model."""
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Sidebar Content
with st.sidebar:
    st.image("https://www.tensorflow.org/images/tf_logo_social.png", width=100)
    st.title("Settings & Info")
    st.info("This application uses a deep learning model trained on the ImageNet dataset (1000 categories).")
    st.write("---")
    st.markdown("### How it works:")
    st.write("1. Upload an image (JPG/PNG).")
    st.write("2. The model extracts features using a CNN.")
    st.write("3. Top predictions are displayed with confidence scores.")

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
        with st.spinner("Model is thinking..."):
            model = load_model()
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                label = label.replace('_', ' ').title()
                score_pct = float(score) * 100
                
                st.write(f"**Rank {i+1}: {label}**")
                st.progress(float(score))
                st.write(f"Confidence: {score_pct:.2f}%")
                st.write("---")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.balloons()
    else:
        st.write("Please upload an image to begin the classification process.")
        st.info("Tip: Try uploading images of animals, vehicles, or common household objects!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with Streamlit & TensorFlow</p>", unsafe_allow_html=True)
