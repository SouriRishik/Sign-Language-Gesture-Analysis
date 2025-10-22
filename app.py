import os
import sys
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} loaded")
except ImportError:
    print('ERROR: TensorFlow not installed')
    sys.exit(1)

# Import MediaPipe with fallback
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    print("MediaPipe available")
except ImportError:
    MP_AVAILABLE = False
    print("MediaPipe not available")

# EXACT SAME CONSTANTS FROM opencv_demo.py
MODEL_PATH = 'cnn_sign_language_model.h5'
MIN_CONFIDENCE = 0.5
HAND_PADDING = 0.35
DEFAULT_LABELS_24 = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

# Global variables
app = Flask(__name__)
model = None
hands = None
target_size = None
labels = None

def load_labels(num_classes: int):
    """Exact same function from opencv_demo.py"""
    if num_classes == 24:
        return DEFAULT_LABELS_24
    return [str(i) for i in range(num_classes)]

def preprocess_roi(bgr: np.ndarray, size: int):
    """Exact same function from opencv_demo.py - simplified"""
    # Convert to grayscale
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    arr = g[..., None]  # Add channel dimension
    return (arr.astype('float32') / 255.0)

def detect_hand_bbox(frame_rgb, hands, w, h, padding=HAND_PADDING):
    """Exact same function from opencv_demo.py"""
    result = hands.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = (max_x - min_x) * padding
    dy = (max_y - min_y) * padding
    min_x = max(0.0, min_x - dx)
    max_x = min(1.0, max_x + dx)
    min_y = max(0.0, min_y - dy)
    max_y = min(1.0, max_y + dy)
    return int(min_x * w), int(min_y * h), int(max_x * w), int(max_y * h)

def initialize_app():
    """Initialize exactly like opencv_demo.py main()"""
    global model, hands, target_size, labels
    
    print("🚀 Initializing Sign Language Recognition...")
    
    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: Model file not found: {MODEL_PATH}')
        return False
    
    # Load model
    print(f'Loading model: {MODEL_PATH}')
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Get model info (same as opencv_demo.py)
    in_shape = model.inputs[0].shape
    if len(in_shape) != 4:
        print('ERROR: Unexpected input shape:', in_shape)
        return False
    
    _, H, W, C = in_shape
    target_size = int(min(H, W))
    num_classes = int(model.outputs[0].shape[-1])
    labels = load_labels(num_classes)
    print(f'Model info: {num_classes} classes -> {labels}')
    print(f'Target size: {target_size}x{target_size}')
    
    # Setup MediaPipe (same as opencv_demo.py)
    if MP_AVAILABLE:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        print('✅ MediaPipe initialized')
        return True
    else:
        print('❌ MediaPipe not available')
        return False

def predict_from_image(image_data):
    """Process image using exact opencv_demo.py logic"""
    try:
        # Decode base64 image
        if 'data:image' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = np.array(image)
        
        # Convert RGB to BGR for OpenCV (same as opencv_demo.py)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Mirror the frame (same as opencv_demo.py DEFAULT_MIRROR=True)
        frame = cv2.flip(frame, 1)
        
        h0, w0 = frame.shape[:2]
        
        # Convert to RGB for MediaPipe (same as opencv_demo.py)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox = detect_hand_bbox(rgb, hands, w0, h0, padding=HAND_PADDING)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1) 
            x2 = min(w0, x2)
            y2 = min(h0, y2)
            
            if x2 > x1 and y2 > y1:
                # Extract ROI and preprocess (exact same as opencv_demo.py)
                roi = frame[y1:y2, x1:x2]
                proc = preprocess_roi(roi, target_size)
                batch = np.expand_dims(proc, 0)
                
                # Predict (same as opencv_demo.py)
                probs = model.predict(batch, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                
                # Get top 3 predictions for display
                top3_indices = np.argsort(probs)[-3:][::-1]
                top3_predictions = [
                    {
                        "letter": labels[i] if i < len(labels) else str(i),
                        "confidence": float(probs[i])
                    }
                    for i in top3_indices
                ]
                
                return {
                    "success": True,
                    "prediction": labels[pred_idx] if pred_idx < len(labels) else str(pred_idx),
                    "confidence": confidence,
                    "top3": top3_predictions,
                    "bbox": [x1, y1, x2, y2],
                    "meets_confidence": confidence >= MIN_CONFIDENCE
                }
            else:
                return {"error": "Invalid bounding box"}
        else:
            return {"error": "No hand detected"}
            
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# Simple HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🤟 ASL Recognition - Clean Version</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            min-height: 100vh;
        }
        .container { 
            max-width: 900px; 
            margin: 0 auto; 
            background: rgba(255,255,255,0.1);
            padding: 20px; 
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        h1 { 
            text-align: center; 
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            align-items: start;
        }
        .video-section {
            text-align: center;
        }
        video { 
            width: 100%; 
            max-width: 500px; 
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transform: scaleX(-1);
        }
        canvas { display: none; }
        .controls { 
            margin: 15px 0; 
        }
        button { 
            padding: 12px 24px; 
            margin: 5px; 
            border: none; 
            border-radius: 25px; 
            font-size: 16px; 
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        .start-btn { 
            background: linear-gradient(45deg, #28a745, #20c997); 
            color: white; 
        }
        .stop-btn { 
            background: linear-gradient(45deg, #dc3545, #fd7e14); 
            color: white; 
        }
        button:hover { transform: translateY(-2px); }
        .results-panel {
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(5px);
        }
        .status { 
            text-align: center; 
            font-size: 1.1em; 
            margin-bottom: 15px;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }
        .prediction-display {
            text-align: center;
            margin: 20px 0;
        }
        .current-letter { 
            font-size: 4em; 
            font-weight: bold; 
            margin: 15px 0;
            color: #feca57;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .confidence { 
            font-size: 1.4em; 
            margin: 10px 0; 
        }
        .confidence.good { color: #28a745; }
        .confidence.poor { color: #ff6b6b; }
        .top3 { margin-top: 20px; }
        .top3 h4 { 
            text-align: center; 
            color: #feca57; 
            margin-bottom: 15px;
        }
        .pred-item { 
            display: flex; 
            justify-content: space-between; 
            padding: 8px 12px; 
            margin: 5px 0; 
            background: rgba(255,255,255,0.1);
            border-radius: 6px;
            border-left: 3px solid transparent;
        }
        .pred-item.rank-1 { border-left-color: #feca57; }
        .pred-item.rank-2 { border-left-color: #ff9f43; }
        .pred-item.rank-3 { border-left-color: #54a0ff; }
        .error { 
            color: #ff6b6b; 
            background: rgba(255,107,107,0.2);
            padding: 15px; 
            border-radius: 8px; 
            margin: 10px 0;
            text-align: center;
        }
        .no-hand {
            text-align: center;
            color: #ff9f43;
            font-style: italic;
            margin: 20px 0;
        }
        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤟 Sign Language Recognition</h1>
        <p style="text-align: center; margin-bottom: 20px; color: #feca57;">
            Using exact opencv_demo.py detection logic
        </p>
        
        <div class="main-grid">
            <div class="video-section">
                <video id="video" autoplay playsinline muted></video>
                <canvas id="canvas"></canvas>
                
                <div class="controls">
                    <button id="startBtn" class="start-btn" onclick="startPrediction()">
                        🎯 Start Recognition
                    </button>
                    <button id="stopBtn" class="stop-btn" onclick="stopPrediction()" style="display: none;">
                        ⏹️ Stop
                    </button>
                </div>
            </div>
            
            <div class="results-panel">
                <div id="status" class="status">
                    📷 Camera ready - Click Start
                </div>
                
                <div id="prediction-display" class="prediction-display" style="display: none;">
                    <div id="current-letter" class="current-letter">-</div>
                    <div id="confidence" class="confidence">0%</div>
                </div>
                
                <div id="no-hand" class="no-hand" style="display: none;">
                    👋 Show your hand to camera
                </div>
                
                <div id="top3" class="top3" style="display: none;">
                    <h4>📊 Live Predictions</h4>
                    <div id="top3-list"></div>
                </div>
                
                <div id="error" class="error" style="display: none;"></div>
            </div>
        </div>
        
        <div style="margin-top: 20px; text-align: center; font-size: 0.9em; opacity: 0.8;">
            <p><strong>Letters:</strong> A-Y (excluding J, Z) • <strong>Confidence:</strong> ≥50% for green</p>
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
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    } 
                });
                video.srcObject = stream;
                
                video.addEventListener('loadedmetadata', () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                });
                
            } catch (err) {
                showError('Camera access denied. Please allow camera and refresh.');
            }
        }
        
        function showError(message) {
            document.getElementById('error').textContent = message;
            document.getElementById('error').style.display = 'block';
        }
        
        function updateStatus(message) {
            document.getElementById('status').textContent = message;
        }
        
        function startPrediction() {
            predictionActive = true;
            document.getElementById('startBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'inline-block';
            document.getElementById('prediction-display').style.display = 'block';
            document.getElementById('top3').style.display = 'block';
            updateStatus('🔄 Recognition active');
            
            // Predict every 600ms for smooth real-time feel
            predictionInterval = setInterval(predict, 600);
        }
        
        function stopPrediction() {
            predictionActive = false;
            clearInterval(predictionInterval);
            document.getElementById('startBtn').style.display = 'inline-block';
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('prediction-display').style.display = 'none';
            document.getElementById('top3').style.display = 'none';
            document.getElementById('no-hand').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            updateStatus('⏸️ Recognition stopped');
        }
        
        async function predict() {
            if (!predictionActive) return;
            
            try {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({image: imageData}),
                    signal: AbortSignal.timeout(3000) // 3 second timeout
                });
                
                const result = await response.json();
                
                if (result.error) {
                    showNoHand();
                } else if (result.success) {
                    showPrediction(result);
                }
                
            } catch (error) {
                console.error('Prediction error:', error);
                if (error.name !== 'AbortError') {
                    showError('Connection error');
                }
            }
        }
        
        function showNoHand() {
            document.getElementById('prediction-display').style.display = 'none';
            document.getElementById('no-hand').style.display = 'block';
            document.getElementById('error').style.display = 'none';
        }
        
        function showPrediction(result) {
            document.getElementById('prediction-display').style.display = 'block';
            document.getElementById('no-hand').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            // Update main prediction
            document.getElementById('current-letter').textContent = result.prediction;
            const confidenceEl = document.getElementById('confidence');
            confidenceEl.textContent = `${(result.confidence * 100).toFixed(1)}%`;
            confidenceEl.className = `confidence ${result.meets_confidence ? 'good' : 'poor'}`;
            
            // Update top 3
            const top3List = document.getElementById('top3-list');
            top3List.innerHTML = '';
            
            result.top3.forEach((pred, index) => {
                const item = document.createElement('div');
                item.className = `pred-item rank-${index + 1}`;
                item.innerHTML = `
                    <span><strong>${pred.letter}</strong></span>
                    <span>${(pred.confidence * 100).toFixed(1)}%</span>
                `;
                top3List.appendChild(item);
            });
        }
        
        // Initialize on page load
        initCamera();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({"error": "No image data"}), 400
        
        result = predict_from_image(image_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "mediapipe_loaded": hands is not None,
        "using_opencv_demo_logic": True
    })

if __name__ == '__main__':
    print("🚀 Starting Clean ASL Recognition Server...")
    
    if initialize_app():
        print("✅ Server ready!")
        print("🌐 Open: http://localhost:5000")
        print("📱 Mobile: http://YOUR_IP:5000")
        print("🎯 Uses EXACT opencv_demo.py logic!")
        
        # For deployment, use production settings
        debug_mode = os.environ.get('FLASK_ENV') != 'production'
        host = '0.0.0.0' if not debug_mode else '127.0.0.1'
        
        app.run(debug=debug_mode, host=host, port=int(os.environ.get('PORT', 5000)))
    else:
        print("❌ Failed to initialize")
        sys.exit(1)