import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="👋",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    return tf.keras.models.load_model('cnn_sign_language_model.h5')

@st.cache_resource
def initialize_mediapipe():
    """Initialize MediaPipe hands"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    return hands, mp_hands

def preprocess_roi(bgr, size=28):
    """Preprocess ROI for model prediction"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype('float32') / 255.0
    return normalized

def detect_hand_bbox(image, hands, padding=0.35):
    """Detect hand bounding box"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    h, w = image.shape[:2]
    landmarks = results.multi_hand_landmarks[0]
    
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    dx = (max_x - min_x) * padding
    dy = (max_y - min_y) * padding
    
    x1 = max(0, int((min_x - dx) * w))
    y1 = max(0, int((min_y - dy) * h))
    x2 = min(w, int((max_x + dx) * w))
    y2 = min(h, int((max_y + dy) * h))
    
    return x1, y1, x2, y2

def predict_sign(image, model, hands):
    """Predict sign language letter"""
    labels = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    
    bbox = detect_hand_bbox(image, hands)
    if bbox is None:
        return None, None, None
    
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None, None, None
    
    roi = image[y1:y2, x1:x2]
    processed = preprocess_roi(roi)
    batch = np.expand_dims(processed[..., None], 0)
    
    predictions = model.predict(batch, verbose=0)[0]
    pred_idx = np.argmax(predictions)
    confidence = predictions[pred_idx]
    
    return labels[pred_idx], confidence, bbox

# Main app
def main():
    st.title("🤟 Sign Language Recognition System")
    st.markdown("Upload an image or use your webcam to recognize ASL letters!")
    
    # Load model and MediaPipe
    try:
        model = load_model()
        hands, mp_hands = initialize_mediapipe()
        st.success("✅ Model and MediaPipe loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Sidebar
    st.sidebar.header("Options")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Image", "Webcam (Live)", "Sample Images"]
    )
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                st.error("Please upload a color image")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Prediction")
                
                # Predict
                prediction, confidence, bbox = predict_sign(image_bgr, model, hands)
                
                if prediction is not None:
                    st.success(f"**Predicted Letter: {prediction}**")
                    st.info(f"Confidence: {confidence*100:.1f}%")
                    
                    # Draw bounding box
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        image_with_box = image_np.copy()
                        cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image_with_box, f"{prediction} ({confidence*100:.1f}%)", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        st.image(image_with_box, use_column_width=True)
                else:
                    st.warning("❌ No hand detected in the image")
    
    elif input_method == "Webcam (Live)":
        st.info("📹 Webcam feature requires running the desktop application")
        st.markdown("To use live webcam recognition:")
        st.code("python opencv_demo.py", language="bash")
        
        st.markdown("### Download Desktop App")
        with open("opencv_demo.py", "r") as f:
            st.download_button(
                label="📥 Download opencv_demo.py",
                data=f.read(),
                file_name="opencv_demo.py",
                mime="text/plain"
            )
    
    elif input_method == "Sample Images":
        st.subheader("📸 Try with Sample Images")
        st.info("Upload some sample ASL images to test the model")
        
        # You can add sample images here
        sample_cols = st.columns(3)
        with sample_cols[0]:
            st.markdown("**Letter A**")
            # Add sample image if available
        with sample_cols[1]:
            st.markdown("**Letter B**")
            # Add sample image if available
        with sample_cols[2]:
            st.markdown("**Letter C**")
            # Add sample image if available
    
    # Footer
    st.markdown("---")
    st.markdown("**Model Info:** 24 ASL Letters (A-Y, excluding J and Z)")
    st.markdown("**Accuracy:** ~83% on test dataset")

if __name__ == "__main__":
    main()