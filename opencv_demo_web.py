import collections
import os
import time
import sys
from typing import List
import base64
from io import BytesIO
from PIL import Image

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

try:
    import tensorflow as tf
except ImportError:
    print('[ERROR] TensorFlow not installed. Install: pip install tensorflow')
    sys.exit(1)

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# EXACT SAME CONSTANTS FROM opencv_demo.py
MODEL_PATH = 'cnn_sign_language_model.h5'
SMOOTH_WINDOW = 3
MIN_CONFIDENCE = 0.5
DEFAULT_MIRROR = True
DETECTION_INTERVAL = 2
INFERENCE_INTERVAL = 2
DETECTION_DOWNSCALE = 1.0
HAND_PADDING = 0.35

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
ENABLE_GPU_MEMORY_GROWTH = True

DEFAULT_LABELS_24 = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

# Flask app
app = Flask(__name__)

# Global variables
model = None
hands = None
mp_hands = None
target_size = None
grayscale = None
labels = None

# EXACT SAME FUNCTIONS FROM opencv_demo.py
def load_labels(num_classes: int) -> List[str]:
    if num_classes == 24:
        return DEFAULT_LABELS_24
    return [str(i) for i in range(num_classes)]

def majority_vote(buf: collections.deque) -> int:
    if not buf:
        return -1
    return collections.Counter(buf).most_common(1)[0][0]

def preprocess_roi(bgr: np.ndarray, size: int, channels: int, grayscale: bool) -> np.ndarray:
    if grayscale:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        if channels == 1:
            arr = g[..., None]
        else:
            arr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return (arr.astype('float32') / 255.0)

def detect_hand_bbox(frame_rgb, hands, w, h, padding=HAND_PADDING):
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
    min_x = max(0.0, min_x - dx); max_x = min(1.0, max_x + dx)
    min_y = max(0.0, min_y - dy); max_y = min(1.0, max_y + dy)
    return int(min_x * w), int(min_y * h), int(max_x * w), int(max_y * h)

def initialize_app():
    """Initialize the same way as opencv_demo.py main()"""
    global model, hands, mp_hands, target_size, grayscale, labels
    
    if not os.path.exists(MODEL_PATH):
        print(f'[ERROR] Model file not found: {MODEL_PATH}')
        return False
    
    if ENABLE_GPU_MEMORY_GROWTH:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            if gpus:
                print(f'[INFO] Enabled GPU memory growth for {len(gpus)} GPU(s)')
        except Exception as e:
            print(f'[WARN] GPU memory growth not set: {e}')

    print(f'[INFO] Loading model {MODEL_PATH}')
    model = tf.keras.models.load_model(MODEL_PATH)
    in_shape = model.inputs[0].shape
    if len(in_shape) != 4:
        print('[ERROR] Unexpected input shape:', in_shape)
        return False
    
    _, H, W, C = in_shape
    target_size = int(min(H, W))
    grayscale = (int(C) == 1)
    num_classes = int(model.outputs[0].shape[-1])
    labels = load_labels(num_classes)
    print(f'[INFO] Classes: {num_classes} -> {labels}')

    if MP_AVAILABLE:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print('[INFO] MediaPipe enabled')
        return True
    else:
        print('[ERROR] MediaPipe not available')
        return False

def predict_from_image(image_data):
    """Process image exactly like opencv_demo.py"""
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
        else:
            return {"error": "Invalid image format"}
        
        # Mirror the frame (same as opencv_demo.py with DEFAULT_MIRROR=True)
        if DEFAULT_MIRROR:
            frame = cv2.flip(frame, 1)
        
        h0, w0 = frame.shape[:2]
        
        # Convert to RGB for MediaPipe (same as opencv_demo.py)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox = detect_hand_bbox(rgb, hands, w0, h0, padding=HAND_PADDING)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w0, x2); y2 = min(h0, y2)
            
            if x2 > x1 and y2 > y1:
                # Extract ROI and preprocess (exact same as opencv_demo.py)
                roi = frame[y1:y2, x1:x2]
                proc = preprocess_roi(roi, target_size, 1, grayscale=True)
                batch = np.expand_dims(proc, 0)
                
                # Predict (same as opencv_demo.py)
                probs = model.predict(batch, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                
                # Get top 3 predictions
                top3_indices = np.argsort(probs)[-3:][::-1]
                top3_predictions = [
                    {"letter": labels[i], "confidence": float(probs[i])}
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
        return {"error": f"Processing error: {str(e)}"}

# HTML template with exact same UI as before but using opencv_demo.py logic
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🤟 Sign Language Recognition (opencv_demo.py Logic)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
            color: #feca57;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 30px;
            align-items: start;
        }
        .camera-section {
            position: relative;
        }
        .video-container {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        video {
            width: 100%;
            max-width: 640px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transform: scaleX(-1);
        }
        .overlay-canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            border-radius: 10px;
            transform: scaleX(-1);
        }
        canvas {
            display: none;
        }
        .controls {
            margin: 20px 0;
            text-align: center;
        }
        button {
            border: none;
            color: white;
            padding: 15px 30px;
            margin: 10px;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .start-btn {
            background: linear-gradient(45deg, #28a745, #20c997);
        }
        .stop-btn {
            background: linear-gradient(45deg, #dc3545, #fd7e14);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        button:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        .predictions-panel {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(5px);
            min-height: 400px;
        }
        .status {
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
        }
        .current-prediction {
            text-align: center;
            margin: 20px 0;
        }
        .prediction-letter {
            font-size: 4em;
            font-weight: bold;
            margin: 10px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            color: #feca57;
        }
        .confidence {
            font-size: 1.3em;
            margin: 10px 0;
        }
        .confidence.good { color: #28a745; }
        .confidence.poor { color: #ff6b6b; }
        .top3-predictions {
            margin-top: 30px;
        }
        .top3-title {
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #feca57;
        }
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 15px;
            margin: 8px 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            border-left: 4px solid transparent;
        }
        .prediction-item.rank-1 { border-left-color: #feca57; }
        .prediction-item.rank-2 { border-left-color: #ff9ff3; }
        .prediction-item.rank-3 { border-left-color: #54a0ff; }
        .letter {
            font-size: 1.8em;
            font-weight: bold;
        }
        .percentage {
            font-size: 1.1em;
            font-weight: bold;
        }
        .error {
            color: #ff6b6b;
            background: rgba(255, 107, 107, 0.2);
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
        @media (max-width: 800px) {
            .main-content {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            .container {
                padding: 15px;
            }
            h1 {
                font-size: 2em;
            }
            .prediction-letter {
                font-size: 3em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤟 Sign Language Recognition</h1>
        <div class="subtitle">Using exact opencv_demo.py detection logic</div>
        
        <div class="main-content">
            <div class="camera-section">
                <div class="video-container">
                    <video id="video" autoplay playsinline></video>
                    <canvas id="overlay" class="overlay-canvas"></canvas>
                </div>
                <canvas id="canvas"></canvas>
                
                <div class="controls">
                    <button id="startBtn" class="start-btn" onclick="startPrediction()">🎯 Start Real-time Prediction</button>
                    <button id="stopBtn" class="stop-btn" onclick="stopPrediction()" style="display: none;">⏹️ Stop Prediction</button>
                </div>
            </div>
            
            <div class="predictions-panel">
                <div id="status" class="status">
                    📷 Camera ready - Click "Start" to begin
                </div>
                
                <div id="current-prediction" class="current-prediction" style="display: none;">
                    <div id="prediction-letter" class="prediction-letter">-</div>
                    <div id="confidence" class="confidence">0%</div>
                </div>
                
                <div id="no-hand" class="no-hand" style="display: none;">
                    👋 Show your hand to the camera
                </div>
                
                <div id="top3-predictions" class="top3-predictions" style="display: none;">
                    <div class="top3-title">📊 Live Predictions</div>
                    <div id="top3-list"></div>
                </div>
            </div>
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
        
        <div style="margin-top: 30px; text-align: center; font-size: 0.9em; opacity: 0.8;">
            <p><strong>Supported Letters:</strong> A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y</p>
            <p><strong>Real-time Processing:</strong> Using exact opencv_demo.py logic with confidence threshold ≥ 50%</p>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const overlay = document.getElementById('overlay');
        const overlayCtx = overlay.getContext('2d');
        const ctx = canvas.getContext('2d');
        
        let predictionActive = false;
        let predictionInterval = null;
        
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
                    
                    const updateOverlay = () => {
                        overlay.width = video.offsetWidth;
                        overlay.height = video.offsetHeight;
                    };
                    
                    updateOverlay();
                    window.addEventListener('resize', updateOverlay);
                });
                
            } catch (err) {
                console.error('Camera error:', err);
                showError('Camera access denied or not available. Please allow camera access and refresh.');
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
            document.getElementById('current-prediction').style.display = 'block';
            document.getElementById('top3-predictions').style.display = 'block';
            updateStatus('🔄 Real-time prediction active (opencv_demo.py logic)');
            
            predictionInterval = setInterval(predictRealtime, 800);
        }
        
        function stopPrediction() {
            predictionActive = false;
            clearInterval(predictionInterval);
            document.getElementById('startBtn').style.display = 'inline-block';
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('current-prediction').style.display = 'none';
            document.getElementById('top3-predictions').style.display = 'none';
            document.getElementById('no-hand').style.display = 'none';
            updateStatus('⏸️ Prediction stopped');
            
            overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
        }
        
        async function predictRealtime() {
            if (!predictionActive) return;
            
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({image: imageData})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showNoHand();
                } else if (data.success) {
                    displayRealtimeResult(data);
                }
            } catch (error) {
                console.error('Prediction error:', error);
            }
        }
        
        function showNoHand() {
            document.getElementById('current-prediction').style.display = 'none';
            document.getElementById('no-hand').style.display = 'block';
            overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
        }
        
        function displayRealtimeResult(data) {
            document.getElementById('current-prediction').style.display = 'block';
            document.getElementById('no-hand').style.display = 'none';
            
            // Update main prediction
            document.getElementById('prediction-letter').textContent = data.prediction;
            const confidenceEl = document.getElementById('confidence');
            confidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
            confidenceEl.className = `confidence ${data.meets_confidence ? 'good' : 'poor'}`;
            
            // Update top 3
            const top3List = document.getElementById('top3-list');
            top3List.innerHTML = '';
            
            data.top3.forEach((pred, index) => {
                const item = document.createElement('div');
                item.className = `prediction-item rank-${index + 1}`;
                item.innerHTML = `
                    <span class="letter">${pred.letter}</span>
                    <span class="percentage">${(pred.confidence * 100).toFixed(1)}%</span>
                `;
                top3List.appendChild(item);
            });
            
            drawBoundingBox(data.bbox, data.meets_confidence);
        }
        
        function drawBoundingBox(bbox, meetsConfidence) {
            overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
            
            if (bbox && bbox.length === 4) {
                let [x1, y1, x2, y2] = bbox;
                
                const scaleX = overlay.width / video.videoWidth;
                const scaleY = overlay.height / video.videoHeight;
                
                x1 = x1 * scaleX;
                y1 = y1 * scaleY;
                x2 = x2 * scaleX;
                y2 = y2 * scaleY;
                
                // Color based on confidence (same as opencv_demo.py)
                const color = meetsConfidence ? '#28a745' : '#dc3545';
                
                overlayCtx.strokeStyle = color;
                overlayCtx.lineWidth = 4;
                overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                
                // Corner markers
                const cornerSize = 25;
                overlayCtx.fillStyle = color;
                overlayCtx.lineWidth = 6;
                
                // Draw corners
                overlayCtx.fillRect(x1 - 3, y1 - 3, cornerSize, 6);
                overlayCtx.fillRect(x1 - 3, y1 - 3, 6, cornerSize);
                overlayCtx.fillRect(x2 - cornerSize + 3, y1 - 3, cornerSize, 6);
                overlayCtx.fillRect(x2 - 3, y1 - 3, 6, cornerSize);
                overlayCtx.fillRect(x1 - 3, y2 - cornerSize + 3, cornerSize, 6);
                overlayCtx.fillRect(x1 - 3, y2 - 3, 6, cornerSize);
                overlayCtx.fillRect(x2 - cornerSize + 3, y2 - cornerSize + 3, cornerSize, 6);
                overlayCtx.fillRect(x2 - 3, y2 - cornerSize + 3, 6, cornerSize);
                
                overlayCtx.font = 'bold 16px Arial';
                overlayCtx.fillStyle = color;
                overlayCtx.fillText('HAND DETECTED', x1, y1 - 10);
            }
        }
        
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
            return jsonify({"error": "No image data provided"}), 400
        
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
    print("🚀 Starting Sign Language Recognition Server (opencv_demo.py logic)...")
    
    if initialize_app():
        print("✅ Server ready!")
        print("📱 Open http://localhost:5000 on your phone or laptop")
        print("🖥️  Or visit http://127.0.0.1:5000")
        print("🎯 Uses EXACT same detection logic as opencv_demo.py!")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize. Check model file and dependencies.")