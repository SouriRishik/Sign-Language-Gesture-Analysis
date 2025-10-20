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

# Flask app for deployment
app = Flask(__name__)

# Disable TensorFlow warnings for cleaner deployment logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

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
    """Initialize the same way as opencv_demo.py main() - optimized for deployment"""
    global model, hands, mp_hands, target_size, grayscale, labels
    
    print(f'[INFO] Starting initialization...')
    print(f'[INFO] Current working directory: {os.getcwd()}')
    print(f'[INFO] Model path: {MODEL_PATH}')
    print(f'[INFO] Model file exists: {os.path.exists(MODEL_PATH)}')
    
    if not os.path.exists(MODEL_PATH):
        print(f'[ERROR] Model file not found: {MODEL_PATH}')
        print(f'[INFO] Files in current directory: {os.listdir(".")}')
        return False
    
    # GPU setup for deployment (safer error handling)
    print(f'[INFO] Setting up TensorFlow...')
    if ENABLE_GPU_MEMORY_GROWTH:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            if gpus:
                print(f'[INFO] Enabled GPU memory growth for {len(gpus)} GPU(s)')
            else:
                print(f'[INFO] No GPUs found, using CPU')
        except Exception as e:
            print(f'[INFO] Running on CPU (GPU setup failed): {e}')

    print(f'[INFO] Loading model {MODEL_PATH}')
    try:
        # Add more verbose loading
        print(f'[INFO] Model file size: {os.path.getsize(MODEL_PATH)} bytes')
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f'[INFO] Model loaded successfully!')
    except Exception as e:
        print(f'[ERROR] Failed to load model: {e}')
        print(f'[ERROR] Exception type: {type(e)}')
        import traceback
        traceback.print_exc()
        return False
        
    try:
        in_shape = model.inputs[0].shape
        print(f'[INFO] Model input shape: {in_shape}')
        if len(in_shape) != 4:
            print('[ERROR] Unexpected input shape:', in_shape)
            return False
        
        _, H, W, C = in_shape
        target_size = int(min(H, W))
        grayscale = (int(C) == 1)
        num_classes = int(model.outputs[0].shape[-1])
        labels = load_labels(num_classes)
        print(f'[INFO] Model setup complete - Classes: {num_classes}, Target size: {target_size}, Grayscale: {grayscale}')
    except Exception as e:
        print(f'[ERROR] Model configuration failed: {e}')
        return False

    # MediaPipe initialization with better error handling
    print(f'[INFO] Setting up MediaPipe...')
    print(f'[INFO] MediaPipe available: {MP_AVAILABLE}')
    
    if MP_AVAILABLE:
        try:
            print(f'[INFO] Importing MediaPipe solutions...')
            mp_hands = mp.solutions.hands
            print(f'[INFO] Creating MediaPipe Hands instance...')
            hands = mp_hands.Hands(
                static_image_mode=False, 
                max_num_hands=1,
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5,
                model_complexity=0  # Use lighter model for deployment
            )
            print('[INFO] MediaPipe enabled successfully!')
            return True
        except Exception as e:
            print(f'[ERROR] MediaPipe initialization failed: {e}')
            print(f'[ERROR] Exception type: {type(e)}')
            import traceback
            traceback.print_exc()
            # Don't fail completely - try to continue without MediaPipe
            print('[WARN] Continuing without MediaPipe - hand detection disabled')
            hands = None
            return True
    else:
        print('[ERROR] MediaPipe not available - hand detection disabled')
        hands = None
        return True

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
        
        # Check if MediaPipe is available
        if hands is None:
            return {"error": "MediaPipe not available - hand detection disabled"}
        
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

# HTML template optimized for mobile deployment
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🤟 ASL Recognition - Deploy Version</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Real-time American Sign Language recognition using AI">
    <meta name="keywords" content="ASL, sign language, AI, recognition, accessibility">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🤟</text></svg>">
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            overflow-x: hidden;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            font-size: 2.2em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            margin-bottom: 25px;
            font-size: 1em;
            color: #feca57;
            opacity: 0.9;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            align-items: start;
        }
        @media (min-width: 900px) {
            .main-content {
                grid-template-columns: 1fr 320px;
                gap: 30px;
            }
        }
        .camera-section {
            position: relative;
        }
        .video-container {
            position: relative;
            display: inline-block;
            width: 100%;
            max-width: 640px;
            margin: 0 auto;
            display: block;
        }
        video {
            width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transform: scaleX(-1);
            background: #000;
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
        canvas { display: none; }
        .controls {
            margin: 15px 0;
            text-align: center;
        }
        button {
            border: none;
            color: white;
            padding: 12px 25px;
            margin: 8px;
            border-radius: 25px;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            touch-action: manipulation;
        }
        .start-btn {
            background: linear-gradient(45deg, #28a745, #20c997);
        }
        .stop-btn {
            background: linear-gradient(45deg, #dc3545, #fd7e14);
        }
        button:hover, button:active {
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
        }
        .status {
            text-align: center;
            font-size: 1em;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
        }
        .current-prediction {
            text-align: center;
            margin: 15px 0;
        }
        .prediction-letter {
            font-size: 3.5em;
            font-weight: bold;
            margin: 10px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            color: #feca57;
        }
        .confidence {
            font-size: 1.2em;
            margin: 10px 0;
        }
        .confidence.good { color: #28a745; }
        .confidence.poor { color: #ff6b6b; }
        .top3-predictions {
            margin-top: 20px;
        }
        .top3-title {
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 12px;
            color: #feca57;
        }
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            margin: 6px 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            border-left: 4px solid transparent;
        }
        .prediction-item.rank-1 { border-left-color: #feca57; }
        .prediction-item.rank-2 { border-left-color: #ff9ff3; }
        .prediction-item.rank-3 { border-left-color: #54a0ff; }
        .letter {
            font-size: 1.6em;
            font-weight: bold;
        }
        .percentage {
            font-size: 1em;
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
            margin: 15px 0;
        }
        .footer-info {
            margin-top: 25px;
            text-align: center;
            font-size: 0.85em;
            opacity: 0.8;
            line-height: 1.4;
        }
        .footer-info p {
            margin: 8px 0;
        }
        @media (max-width: 600px) {
            .container { padding: 15px; }
            h1 { font-size: 1.8em; }
            .prediction-letter { font-size: 3em; }
            button { padding: 10px 20px; font-size: 14px; }
        }
        /* Loading animation */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤟 ASL Recognition</h1>
        <div class="subtitle">Real-time American Sign Language detection - Accessible anywhere!</div>
        
        <!-- Debug Section -->
        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3>🔧 Debug Info</h3>
            <div id="debug-info">
                <p>📍 <strong>Location:</strong> <span id="location-info">Checking...</span></p>
                <p>🔒 <strong>HTTPS:</strong> <span id="https-info">Checking...</span></p>
                <p>📷 <strong>Camera API:</strong> <span id="camera-api-info">Checking...</span></p>
                <p>🤖 <strong>Backend:</strong> <span id="backend-info">Checking...</span></p>
                <button onclick="testBackend()" style="background: #28a745; border: none; color: white; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px 5px;">Test Backend</button>
                <button onclick="testCamera()" style="background: #007bff; border: none; color: white; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px 5px;">Test Camera</button>
            </div>
        </div>
        
        <div class="main-content">
            <div class="camera-section">
                <div class="video-container">
                    <video id="video" autoplay playsinline muted></video>
                    <canvas id="overlay" class="overlay-canvas"></canvas>
                </div>
                <canvas id="canvas"></canvas>
                
                <div class="controls">
                    <button id="startBtn" class="start-btn" onclick="startPrediction()">🎯 Start Recognition</button>
                    <button id="stopBtn" class="stop-btn" onclick="stopPrediction()" style="display: none;">⏹️ Stop</button>
                </div>
            </div>
            
            <div class="predictions-panel">
                <div id="status" class="status">
                    📷 Camera ready - Tap "Start" to begin
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
        
        <div class="footer-info">
            <p><strong>Supported Letters:</strong> A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y</p>
            <p><strong>AI-Powered:</strong> Real-time detection with 83%+ accuracy using opencv_demo.py logic</p>
            <p><strong>Accessible:</strong> Works on phones, tablets, and computers worldwide</p>
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
        let retryCount = 0;
        const maxRetries = 3;
        
        async function initCamera() {
            try {
                updateStatus('🔄 Requesting camera access...');
                
                const constraints = { 
                    video: { 
                        width: { ideal: 640, max: 1280 },
                        height: { ideal: 480, max: 720 },
                        facingMode: 'user'
                    } 
                };
                
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
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
                    
                    updateStatus('📷 Camera ready - Tap "Start" to begin');
                });
                
                video.addEventListener('error', (e) => {
                    console.error('Video error:', e);
                    showError('Camera error occurred. Please refresh and try again.');
                });
                
            } catch (err) {
                console.error('Camera error:', err);
                let errorMessage = 'Camera access denied. ';
                
                if (err.name === 'NotAllowedError') {
                    errorMessage += 'Please allow camera access and refresh the page.';
                } else if (err.name === 'NotFoundError') {
                    errorMessage += 'No camera found on this device.';
                } else {
                    errorMessage += 'Please check camera permissions and refresh.';
                }
                
                showError(errorMessage);
            }
        }
        
        function showError(message) {
            document.getElementById('error').textContent = message;
            document.getElementById('error').style.display = 'block';
            setTimeout(() => {
                document.getElementById('error').style.display = 'none';
            }, 8000);
        }
        
        function updateStatus(message) {
            document.getElementById('status').innerHTML = message;
        }
        
        function startPrediction() {
            predictionActive = true;
            document.getElementById('startBtn').style.display = 'none';
            document.getElementById('stopBtn').style.display = 'inline-block';
            document.getElementById('current-prediction').style.display = 'block';
            document.getElementById('top3-predictions').style.display = 'block';
            updateStatus('🔄 <span class="loading"></span> AI processing...');
            
            retryCount = 0;
            predictionInterval = setInterval(predictRealtime, 900);
        }
        
        function stopPrediction() {
            predictionActive = false;
            clearInterval(predictionInterval);
            document.getElementById('startBtn').style.display = 'inline-block';
            document.getElementById('stopBtn').style.display = 'none';
            document.getElementById('current-prediction').style.display = 'none';
            document.getElementById('top3-predictions').style.display = 'none';
            document.getElementById('no-hand').style.display = 'none';
            updateStatus('⏸️ Recognition stopped');
            
            overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
        }
        
        async function predictRealtime() {
            if (!predictionActive) return;
            
            try {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.7);
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({image: imageData}),
                    timeout: 5000
                });
                
                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }
                
                const data = await response.json();
                
                retryCount = 0; // Reset on success
                
                if (data.error) {
                    showNoHand();
                    updateStatus('🤖 AI ready - Show your hand');
                } else if (data.success) {
                    displayRealtimeResult(data);
                    updateStatus('🎯 Live recognition active');
                }
                
            } catch (error) {
                console.error('Prediction error:', error);
                retryCount++;
                
                if (retryCount >= maxRetries) {
                    showError('Connection issue. Please check internet and refresh.');
                    stopPrediction();
                } else {
                    updateStatus(`🔄 Reconnecting... (${retryCount}/${maxRetries})`);
                }
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
                
                // Color based on confidence
                const color = meetsConfidence ? '#28a745' : '#dc3545';
                
                overlayCtx.strokeStyle = color;
                overlayCtx.lineWidth = 3;
                overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                
                // Corner markers
                const cornerSize = 20;
                overlayCtx.fillStyle = color;
                overlayCtx.lineWidth = 5;
                
                // Draw corners
                overlayCtx.fillRect(x1 - 2, y1 - 2, cornerSize, 4);
                overlayCtx.fillRect(x1 - 2, y1 - 2, 4, cornerSize);
                overlayCtx.fillRect(x2 - cornerSize + 2, y1 - 2, cornerSize, 4);
                overlayCtx.fillRect(x2 - 2, y1 - 2, 4, cornerSize);
                overlayCtx.fillRect(x1 - 2, y2 - cornerSize + 2, cornerSize, 4);
                overlayCtx.fillRect(x1 - 2, y2 - 2, 4, cornerSize);
                overlayCtx.fillRect(x2 - cornerSize + 2, y2 - cornerSize + 2, cornerSize, 4);
                overlayCtx.fillRect(x2 - 2, y2 - cornerSize + 2, 4, cornerSize);
                
                overlayCtx.font = 'bold 14px Arial';
                overlayCtx.fillStyle = color;
                overlayCtx.fillText('HAND', x1, y1 - 8);
            }
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', () => {
            initDebugInfo();
            initCamera();
        });
        
        // Debug functions
        function initDebugInfo() {
            // Check location
            document.getElementById('location-info').textContent = window.location.href;
            
            // Check HTTPS
            document.getElementById('https-info').textContent = window.location.protocol === 'https:' ? '✅ Secure' : '❌ Not Secure (HTTP)';
            
            // Check Camera API
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                document.getElementById('camera-api-info').textContent = '✅ Supported';
            } else {
                document.getElementById('camera-api-info').textContent = '❌ Not Supported';
            }
        }
        
        async function testBackend() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                document.getElementById('backend-info').innerHTML = `✅ Backend OK - Model: ${data.model_loaded ? '✅' : '❌'} MediaPipe: ${data.mediapipe_loaded ? '✅' : '❌'}`;
            } catch (error) {
                document.getElementById('backend-info').textContent = `❌ Backend Error: ${error.message}`;
            }
        }
        
        async function testCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                document.getElementById('camera-api-info').textContent = '✅ Camera Access Granted';
                // Stop the stream
                stream.getTracks().forEach(track => track.stop());
            } catch (error) {
                document.getElementById('camera-api-info').textContent = `❌ Camera Error: ${error.name}`;
            }
        }
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && predictionActive) {
                stopPrediction();
            }
        });
        
        // Handle mobile orientation changes
        window.addEventListener('orientationchange', () => {
            setTimeout(() => {
                const updateOverlay = () => {
                    overlay.width = video.offsetWidth;
                    overlay.height = video.offsetHeight;
                };
                updateOverlay();
            }, 500);
        });
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
        "mediapipe_available": MP_AVAILABLE,
        "deployment_ready": True,
        "using_opencv_demo_logic": True,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "current_directory": os.getcwd(),
        "python_version": sys.version,
        "tensorflow_version": tf.__version__
    })

@app.route('/api/info')
def api_info():
    return jsonify({
        "name": "ASL Recognition API",
        "version": "1.0.0",
        "supported_letters": len(labels) if labels else 24,
        "confidence_threshold": MIN_CONFIDENCE,
        "model_accuracy": "83%+",
        "real_time": True
    })

# Error handlers for deployment
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting ASL Recognition Server (Deployment Version)...")
    
    if initialize_app():
        print("✅ Server ready for deployment!")
        print("🌍 Will be accessible worldwide once deployed")
        print("📱 Mobile & laptop compatible")
        print("🎯 Using EXACT opencv_demo.py logic for maximum accuracy!")
        
        # For deployment, use environment port or default to 5000
        port = int(os.environ.get('PORT', 5000))
        
        # Check if running in production or development
        is_production = os.environ.get('RENDER') or os.environ.get('HEROKU_APP_NAME')
        
        if is_production:
            print("🌐 Running in PRODUCTION mode")
            app.run(debug=False, host='0.0.0.0', port=port)
        else:
            print("💻 Running in DEVELOPMENT mode")
            app.run(debug=True, host='0.0.0.0', port=port)
    else:
        print("❌ Failed to initialize. Check model file and dependencies.")
        sys.exit(1)