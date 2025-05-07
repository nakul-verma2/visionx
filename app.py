from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import os
import numpy as np
from io import BytesIO
import base64
import time
import threading
from glob import glob
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = YOLO('model/best.pt')

video_capture = None
video_thread = None
frame = None
lock = threading.Lock()
is_streaming = False

UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def initialize_webcam():
    """Attempt to initialize webcam with retry logic."""
    global video_capture
    for index in range(3):  # Try indices 0, 1, 2
        video_capture = cv2.VideoCapture(index)
        if video_capture.isOpened():
            logger.debug(f"Webcam opened successfully on index {index}")
            return True
        video_capture.release()
        logger.debug(f"Failed to open webcam on index {index}")
        time.sleep(0.5)
    logger.error("Could not open any webcam")
    return False

def generate_frames():
    """Generate frames for live video streaming with YOLOv8 detection."""
    global frame, is_streaming, video_capture
    logger.debug("Starting frame generation")
    while is_streaming:
        if video_capture is None or not video_capture.isOpened():
            logger.error("Video capture not initialized or failed")
            time.sleep(0.1)
            continue

        ret, current_frame = video_capture.read()
        if not ret:
            logger.error("Failed to capture frame")
            time.sleep(0.1)
            continue

        # Run YOLOv8 inference
        try:
            results = model(current_frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(current_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"YOLOv8 inference failed: {str(e)}")

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if ret:
            with lock:
                frame = buffer.tobytes()
            logger.debug("Frame generated")
        else:
            logger.error("Failed to encode frame")
        time.sleep(0.03)  # Control frame rate (~30 FPS)
    logger.debug("Frame generation stopped")

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and save to uploads folder."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = f"image_{int(time.time())}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.debug(f"Image uploaded: {filename}")
        return jsonify({'message': 'Image uploaded', 'filename': filename}), 200

@app.route('/analyze_image/<filename>', methods=['GET'])
def analyze_image(filename):
    """Analyze uploaded image using YOLOv8 and return the processed image."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        logger.error(f"Image not found: {file_path}")
        return jsonify({'error': 'Image not found'}), 404

    # Read and process image
    img = cv2.imread(file_path)
    results = model.predict(source=img, save=False, conf=0.5)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append({
                'label': label,
                'confidence': float(conf),
                'bbox': {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1},
                'class': label.lower().replace(' ', '-')
            })

        # Draw detections on image
        img_annotated = result.plot()
        img_bgr = cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR)

        # Resize image to fixed size for frontend
        target_width = 640
        target_height = 480
        h, w = img_bgr.shape[:2]
        scaling_factor = min(target_width / w, target_height / h)
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)

        # Pad image to exact size
        top = (target_height - new_size[1]) // 2
        bottom = target_height - new_size[1] - top
        left = (target_width - new_size[0]) // 2
        right = target_width - new_size[0] - left
        img_bgr = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    logger.debug(f"Detections found: {len(detections)}")

    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'detections': detections,
        'image': f'data:image/jpeg;base64,{img_base64}'
    }), 200

@app.route('/start_video', methods=['POST'])
def start_video():
    """Start live video feed."""
    global video_capture, video_thread, is_streaming
    if is_streaming:
        logger.debug("Video already streaming")
        return jsonify({'message': 'Video already streaming'}), 200

    if not initialize_webcam():
        return jsonify({'error': 'Could not open webcam'}), 500

    is_streaming = True
    video_thread = threading.Thread(target=generate_frames)
    video_thread.daemon = True
    video_thread.start()
    logger.debug("Video streaming started")
    return jsonify({'message': 'Video streaming started'}), 200

@app.route('/stop_video', methods=['POST'])
def stop_video():
    """Stop live video feed."""
    global video_capture, is_streaming, video_thread
    if not is_streaming:
        logger.debug("No video streaming")
        return jsonify({'message': 'No video streaming'}), 200

    is_streaming = False
    if video_thread:
        video_thread.join()
        video_thread = None
    if video_capture:
        video_capture.release()
        video_capture = None
    frame = None
    logger.debug("Video streaming stopped")
    return jsonify({'message': 'Video streaming stopped'}), 200

@app.route('/video_feed')
def video_feed():
    """Stream live video feed with detections."""
    logger.debug("Accessing video feed")
    def stream():
        global frame
        while is_streaming:
            with lock:
                if frame is None:
                    logger.debug("No frame available")
                    time.sleep(0.03)
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def serve_index():
    """Serve the main HTML page."""
    return send_file('visionx.html')

@app.route('/explore')
def serve_explore():
    """Serve the explore HTML page."""
    return send_file('explorepage.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)