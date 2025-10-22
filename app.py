import os
import sys
import base64
import gc  # For garbage collection
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, render_template_string

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    print(f"[INFO] TensorFlow version: {tf.__version__}")
    
    # Configure TensorFlow for minimal memory usage
    try:
        # Force CPU usage for memory efficiency on Render
        tf.config.set_visible_devices([], 'GPU')
        
        # Limit parallelism to reduce memory overhead
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        print("[INFO] TensorFlow memory optimization applied")
        
    except Exception as e:
        print(f"[WARN] TensorFlow optimization failed: {e}")
        
except ImportError:
    print('[ERROR] TensorFlow not installed')
    sys.exit(1)

# Import MediaPipe with fallback
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    print("[INFO] MediaPipe available")
except ImportError:
    MP_AVAILABLE = False
    print("[WARN] MediaPipe not available - will use center crop")

# Configuration
MODEL_PATH = 'cnn_sign_language_model.h5'
MIN_CONFIDENCE = 0.5
HAND_PADDING = 0.15  # Reduced for less processing
ASL_LABELS = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

# Disable TensorFlow warnings and optimize for deployment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress more logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.get_logger().setLevel('ERROR')

# Global variables
app = Flask(__name__)
model = None
hands = None
target_size = None

def load_model_safe():
    """Load the .h5 model with multiple compatibility methods"""
    global model, target_size
    
    print(f"[INFO] Loading model: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return False
    
    # Method 1: Try loading normally
    try:
        print("[INFO] Attempting normal model loading...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("[INFO] ✅ Model loaded successfully!")
    except Exception as e1:
        print(f"[WARN] Normal loading failed: {str(e1)[:100]}")
        
        # Method 2: Try with safe_mode=False for compatibility
        try:
            print("[INFO] Attempting with safe_mode=False...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
            print("[INFO] ✅ Model loaded with safe_mode=False!")
        except Exception as e2:
            print(f"[WARN] Safe mode loading failed: {str(e2)[:100]}")
            
            # Method 3: Try with custom objects for batch_shape issue
            try:
                print("[INFO] Attempting with custom objects...")
                custom_objects = {
                    'InputLayer': tf.keras.layers.InputLayer,
                    'Dense': tf.keras.layers.Dense,
                    'Conv2D': tf.keras.layers.Conv2D,
                    'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                    'Flatten': tf.keras.layers.Flatten,
                    'Dropout': tf.keras.layers.Dropout
                }
                model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
                print("[INFO] ✅ Model loaded with custom objects!")
            except Exception as e3:
                print(f"[ERROR] All loading methods failed")
                print(f"[ERROR] Error 1: {str(e1)[:100]}")
                print(f"[ERROR] Error 2: {str(e2)[:100]}")
                print(f"[ERROR] Error 3: {str(e3)[:100]}")
                return False
    
    # Get model info
    try:
        input_shape = model.inputs[0].shape
        output_shape = model.outputs[0].shape
        target_size = int(input_shape[1])  # Assuming square input
        
        print(f"[INFO] Model input shape: {input_shape}")
        print(f"[INFO] Model output shape: {output_shape}")
        print(f"[INFO] Target size: {target_size}")
        print(f"[INFO] Expected classes: {len(ASL_LABELS)}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to get model info: {e}")
        return False

def setup_mediapipe():
    """MediaPipe disabled for memory optimization"""
    global hands
    
    # Skip MediaPipe initialization to save memory on Render
    hands = None
    print("[INFO] MediaPipe disabled for memory optimization - using center crop only")
    return True

def detect_hand_region(image):
    """Use center crop only - MediaPipe disabled for memory optimization"""
    h, w = image.shape[:2]
    
    # Always use center crop to save memory (no MediaPipe processing)
    center_x, center_y = w // 2, h // 2
    size = min(w, h) // 2
    x1 = max(0, center_x - size)
    y1 = max(0, center_y - size)
    x2 = min(w, center_x + size)
    y2 = min(h, center_y + size)
    
    print("[DEBUG] Using center crop (MediaPipe disabled for memory)")
    return (x1, y1, x2, y2)

def preprocess_image(image_region):
    """Preprocess image region for model prediction - optimized for speed"""
    # Convert to grayscale using faster method
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_region
    
    # Resize to target size with fastest interpolation
    resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    # Normalize and reshape in one step
    normalized = (resized.astype('float32') / 255.0)[..., np.newaxis]
    
    # Add batch dimension
    batch = np.expand_dims(normalized, axis=0)
    
    return batch

def predict_sign(image_data):
    """Main prediction function with aggressive memory optimization"""
    try:
        print("[DEBUG] Starting prediction with memory optimization...")
        
        # Force garbage collection at start
        gc.collect()
        
        # Decode base64 image with size optimization
        if 'data:image' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Resize image immediately to reduce processing load (very small)
        pil_image = pil_image.resize((160, 120), Image.LANCZOS)
        image = np.array(pil_image)
        
        print("[DEBUG] Image decoded and resized successfully")
        
        # Clean up immediately and force garbage collection
        del image_bytes, pil_image
        gc.collect()
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return {"error": "Invalid image format"}
        
        # Mirror image (like webcam)
        image = cv2.flip(image, 1)
        
        # Detect hand region (always returns a region now)
        bbox = detect_hand_region(image)
        x1, y1, x2, y2 = bbox
        
        print(f"[DEBUG] Hand region: {bbox}")
        
        # Extract and validate region
        if x2 <= x1 or y2 <= y1:
            return {"error": "Invalid hand region"}
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return {"error": "Empty hand region"}
        
        # Clean up large image array
        del image
        
        # Preprocess for model
        processed = preprocess_image(roi)
        
        # Clean up ROI
        del roi
        
        # Predict with extreme memory optimization
        print("[DEBUG] Starting model prediction...")
        
        # Force garbage collection before prediction
        gc.collect()
        
        # Use the most memory-efficient prediction method
        with tf.device('/CPU:0'):  # Ensure CPU usage
            predictions = model(processed, training=False).numpy()[0]
        
        print("[DEBUG] Model prediction completed")
        
        # Clean up processed array immediately
        del processed
        gc.collect()  # Force cleanup
        
        # Get results efficiently
        pred_idx = np.argmax(predictions)
        confidence = float(predictions[pred_idx])
        predicted_letter = ASL_LABELS[pred_idx] if pred_idx < len(ASL_LABELS) else f"Class_{pred_idx}"
        
        # Get top 3 predictions
        top3_indices = np.argsort(predictions)[-3:][::-1]
        top3_predictions = [
            {
                "letter": ASL_LABELS[i] if i < len(ASL_LABELS) else f"Class_{i}",
                "confidence": float(predictions[i])
            }
            for i in top3_indices
        ]
        
        # Clean up prediction arrays
        del predictions, top3_indices
        
        # Force garbage collection to free memory
        gc.collect()
        
        print("[DEBUG] Prediction completed successfully")
        
        return {
            "success": True,
            "prediction": predicted_letter,
            "confidence": confidence,
            "meets_confidence": confidence >= MIN_CONFIDENCE,
            "top3": top3_predictions,
            "bbox": [x1, y1, x2, y2],
            "detection_method": "mediapipe" if hands else "center_crop"
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Initialize everything when module loads (for Gunicorn)
def initialize():
    """Initialize model and MediaPipe"""
    print("🚀 Initializing ASL Recognition Server...")
    
    # Setup TensorFlow
    try:
        # Limit memory growth
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    
    # Load model
    if not load_model_safe():
        print("❌ Model loading failed!")
        return False
    
    # Setup MediaPipe
    if not setup_mediapipe():
        print("❌ MediaPipe setup failed!")
        return False
    
    print("✅ Initialization complete!")
    return True

# Initialize when imported
initialization_success = initialize()

# Flask routes
@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>🤟 ASL Recognition</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f8ff; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; }
        .video-container { position: relative; display: flex; justify-content: center; margin-left: 40px; }
        video { width: 100%; max-width: 640px; border-radius: 10px; transform: scaleX(-1); }
        canvas { display: none; }
        .controls { margin: 20px 0; text-align: center; }
        button { padding: 12px 24px; margin: 10px; border: none; border-radius: 25px; font-size: 16px; cursor: pointer; }
        .start-btn { background: #4CAF50; color: white; }
        .stop-btn { background: #f44336; color: white; }
        .results { margin-top: 20px; padding: 20px; background: #f9f9f9; border-radius: 10px; }
        .prediction { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .top3-predictions { margin-top: 15px; }
        .prediction-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; margin: 5px 0; background: white; border-radius: 8px; border-left: 4px solid #ddd; }
        .prediction-item.rank-1 { border-left-color: #4CAF50; font-weight: bold; }
        .prediction-item.rank-2 { border-left-color: #FF9800; }
        .prediction-item.rank-3 { border-left-color: #2196F3; }
        .letter { font-size: 18px; font-weight: bold; }
        .percentage { font-size: 14px; color: #666; }
        .confidence { margin: 10px 0; }
        .error { color: red; font-weight: bold; }
        .success { color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤟 ASL Recognition</h1>
        <div class="video-container">
            <video id="video" autoplay playsinline muted></video>
        </div>
        <canvas id="canvas"></canvas>
        
        <div class="controls">
            <button id="startBtn" class="start-btn" onclick="startPrediction()">Start Recognition</button>
            <button id="stopBtn" class="stop-btn" onclick="stopPrediction()" style="display: none;">Stop</button>
        </div>
        
        <div class="results">
            <div id="status">Camera ready - Click Start to begin</div>
            <div id="prediction" class="prediction" style="display: none;"></div>
            <div id="confidence" class="confidence" style="display: none;"></div>
            <div id="top3-predictions" class="top3-predictions" style="display: none;">
                <h4 style="margin: 15px 0 10px 0; color: #333;">📊 Top 3 Predictions:</h4>
                <div id="top3-list"></div>
            </div>
            <div id="error" class="error" style="display: none;"></div>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let predictionActive = false;
        let predictionInterval = null;

        // Initialize camera
        async function initCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480, facingMode: 'user' } 
                });
                video.srcObject = stream;
                
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    document.getElementById('status').textContent = 'Camera ready - Click Start to begin';
                });
            } catch (err) {
                document.getElementById('error').textContent = 'Camera access denied. Please allow camera access.';
                document.getElementById('error').style.display = 'block';
            }
        }

        function startPrediction() {
            predictionActive = true;
            document.getElementById('startBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'inline-block';
            document.getElementById('status').textContent = 'Recognition active...';
            
            predictionInterval = setInterval(predict, 2000); // Predict every 2 seconds for less server load
        }

        function stopPrediction() {
            predictionActive = false;
            clearInterval(predictionInterval);
            document.getElementById('startBtn').style.display = 'inline-block';
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('status').textContent = 'Recognition stopped';
            document.getElementById('prediction').style.display = 'none';
            document.getElementById('confidence').style.display = 'none';
            document.getElementById('top3-predictions').style.display = 'none';
            document.getElementById('error').style.display = 'none';
        }

        async function predict() {
            if (!predictionActive) return;

            try {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);

                // Add timeout to fetch request (reduced for faster feedback)
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData }),
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                const result = await response.json();

                if (result.error) {
                    // Handle "no hand detected" differently from other errors
                    if (result.show_hand_message) {
                        document.getElementById('status').textContent = '👋 Show your hand to the camera';
                        document.getElementById('error').style.display = 'none';
                    } else {
                        document.getElementById('error').textContent = result.error;
                        document.getElementById('error').style.display = 'block';
                    }
                    document.getElementById('prediction').style.display = 'none';
                    document.getElementById('confidence').style.display = 'none';
                } else {
                    document.getElementById('error').style.display = 'none';
                    document.getElementById('status').textContent = 'Recognition active...';
                    
                    // Show main prediction
                    document.getElementById('prediction').textContent = `🏆 Best: ${result.prediction}`;
                    document.getElementById('prediction').style.display = 'block';
                    document.getElementById('confidence').textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
                    document.getElementById('confidence').style.display = 'block';
                    document.getElementById('confidence').className = result.meets_confidence ? 'confidence success' : 'confidence';
                    
                    // Show top 3 predictions
                    if (result.top3 && result.top3.length > 0) {
                        const top3List = document.getElementById('top3-list');
                        top3List.innerHTML = '';
                        
                        result.top3.forEach((pred, index) => {
                            const item = document.createElement('div');
                            item.className = `prediction-item rank-${index + 1}`;
                            item.innerHTML = `
                                <span class="letter">${index + 1}. ${pred.letter}</span>
                                <span class="percentage">${(pred.confidence * 100).toFixed(1)}%</span>
                            `;
                            top3List.appendChild(item);
                        });
                        
                        document.getElementById('top3-predictions').style.display = 'block';
                    } else {
                        document.getElementById('top3-predictions').style.display = 'none';
                    }
                }
            } catch (error) {
                console.error('Prediction error:', error);
                if (error.name === 'AbortError') {
                    document.getElementById('error').textContent = 'Request timed out. Server may be overloaded.';
                } else if (error.message.includes('Failed to fetch')) {
                    document.getElementById('error').textContent = 'Connection lost. Checking server status...';
                } else {
                    document.getElementById('error').textContent = 'Connection error. Please refresh the page.';
                }
                document.getElementById('error').style.display = 'block';
            }
        }

        // Initialize camera on page load
        document.addEventListener('DOMContentLoaded', initCamera);
    </script>
</body>
</html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("[DEBUG] Received prediction request")
        
        if not initialization_success:
            print("[ERROR] Server not properly initialized")
            return jsonify({"error": "Server not properly initialized"}), 500
        
        if model is None:
            print("[ERROR] Model not loaded")
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            print("[ERROR] No image data provided")
            return jsonify({"error": "No image data provided"}), 400
        
        print("[DEBUG] Processing prediction...")
        result = predict_sign(data['image'])
        print("[DEBUG] Returning result")
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Prediction endpoint failed: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy" if initialization_success else "unhealthy",
        "model_loaded": model is not None,
        "mediapipe_available": MP_AVAILABLE,
        "mediapipe_initialized": hands is not None,
        "tensorflow_version": tf.__version__
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)