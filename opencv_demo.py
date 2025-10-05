import collections
import os
import time
import sys
from typing import List

import cv2
import numpy as np

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

MODEL_PATH = 'cnn_sign_language_model.h5'
SMOOTH_WINDOW = 8
MIN_CONFIDENCE = 0.5
TEXT_FG_COLOR = (0, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
FPS_FG_COLOR = (255, 255, 0)

DEFAULT_MIRROR = True
DETECTION_INTERVAL = 2
INFERENCE_INTERVAL = 1
DETECTION_DOWNSCALE = 1.3

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

ENABLE_GPU_MEMORY_GROWTH = True

DEFAULT_LABELS_24 = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']


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


def detect_hand_bbox(frame_rgb, hands, w, h, padding=0.15):
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


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'[ERROR] Model file not found: {MODEL_PATH}')
        return
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
        return
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
        use_mediapipe = True
        print('[INFO] MediaPipe enabled')
    else:
        hands = None
        use_mediapipe = False
        print('[INFO] MediaPipe not installed; using center ROI')

    cap = cv2.VideoCapture(0)
    if CAPTURE_WIDTH > 0 and CAPTURE_HEIGHT > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not cap.isOpened():
        print('[ERROR] Cannot access webcam')
        return

    fallback_roi = None

    mirror = DEFAULT_MIRROR
    pred_buf = collections.deque(maxlen=SMOOTH_WINDOW)
    last_label = ''
    last_conf = 0.0
    last_pred_idx = -1
    last_probs = None
    last_bbox = None
    last_detect_frame = -999
    last_infer_frame = -999
    fps_t = time.time()
    frame_i = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('[WARN] Frame read failed')
                break
            frame_i += 1
            if mirror:
                frame = cv2.flip(frame, 1)

            h0, w0 = frame.shape[:2]

            if use_mediapipe:
                if frame_i - last_detect_frame >= DETECTION_INTERVAL:
                    det_frame = frame
                    if DETECTION_DOWNSCALE > 1.0:
                        ds_w = int(w0 / DETECTION_DOWNSCALE)
                        ds_h = int(h0 / DETECTION_DOWNSCALE)
                        det_small = cv2.resize(frame, (ds_w, ds_h), interpolation=cv2.INTER_LINEAR)
                        rgb = cv2.cvtColor(det_small, cv2.COLOR_BGR2RGB)
                        bbox_small = detect_hand_bbox(rgb, hands, ds_w, ds_h)
                        if bbox_small:
                            x1s, y1s, x2s, y2s = bbox_small
                            scale = DETECTION_DOWNSCALE
                            last_bbox = (int(x1s*scale), int(y1s*scale), int(x2s*scale), int(y2s*scale))
                        else:
                            last_bbox = None
                    else:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        last_bbox = detect_hand_bbox(rgb, hands, w0, h0)
                    last_detect_frame = frame_i
                bbox = last_bbox
            else:
                if fallback_roi is None:
                    side = int(min(w0, h0) * 0.6)
                    x1 = (w0 - side) // 2
                    y1 = (h0 - side) // 2
                    fallback_roi = (x1, y1, x1 + side, y1 + side)
                bbox = fallback_roi

            if bbox:
                x1, y1, x2, y2 = bbox
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w0, x2); y2 = min(h0, y2)
                if x2 > x1 and y2 > y1:
                    if frame_i - last_infer_frame >= INFERENCE_INTERVAL:
                        roi = frame[y1:y2, x1:x2]
                        proc = preprocess_roi(roi, target_size, int(C), grayscale)
                        batch = np.expand_dims(proc, 0)
                        last_probs = model.predict(batch, verbose=0)[0]
                        last_pred_idx = int(np.argmax(last_probs))
                        last_infer_frame = frame_i
                    if last_probs is not None:
                        idx = last_pred_idx
                        conf = float(last_probs[idx])
                        pred_buf.append(idx)
                        voted = majority_vote(pred_buf)
                        voted_conf = conf if voted == idx else last_probs[voted]
                        last_label = labels[voted] if voted < len(labels) else str(voted)
                        last_conf = float(voted_conf)
                        color = (0,200,0) if last_conf >= MIN_CONFIDENCE else (0,0,255)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                else:
                    last_label = ''
            else:
                last_label = ''

            if frame_i % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - fps_t)
                fps_t = now
            else:
                fps = None

            text = f'{last_label} {last_conf*100:.1f}%' if last_label else ('Detecting...' if use_mediapipe else 'Place hand')
            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_BG_COLOR, 4, cv2.LINE_AA)
            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_FG_COLOR, 2, cv2.LINE_AA)
            if fps is not None:
                fps_text = f'FPS: {fps:.1f}'
                cv2.putText(frame, fps_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_BG_COLOR, 3, cv2.LINE_AA)
                cv2.putText(frame, fps_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FPS_FG_COLOR, 1, cv2.LINE_AA)

            cv2.imshow('Sign Inference (CNN)', frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == ord('r') and not use_mediapipe:
                fallback_roi = None
                print('[INFO] ROI reset')
            elif k == ord('f'):
                mirror = not mirror
                print(f'[INFO] Mirror set to {mirror}')
            elif k == ord('d') and use_mediapipe:
                if DETECTION_INTERVAL == 1:
                    print('[INFO] Detection interval already 1 (cannot toggle at runtime without code change)')
                else:
                    print(f'[INFO] Detection interval fixed at {DETECTION_INTERVAL} (change constant and restart)')

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if use_mediapipe:
            hands.close()
        print('[INFO] Exit')


if __name__ == '__main__':
    main()
