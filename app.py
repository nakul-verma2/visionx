from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import os
import numpy as np
import base64
import time
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load YOLO model
model = YOLO('model/best.pt')

# Upload folder - use /tmp for Hugging Face deployment compatibility
UPLOAD_FOLDER = '/tmp/visionx_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
logger.debug(f"Uploads folder set to: {UPLOAD_FOLDER}")

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
        logger.debug(f"Saved at: {file_path}")
        return jsonify({'message': 'Image uploaded', 'filename': filename}), 200

@app.route('/analyze_image/<filename>', methods=['GET'])
def analyze_image(filename):
    """Analyze uploaded image using YOLOv8 and return the processed image."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    logger.debug(f"Analyzing image: {filename}")
    logger.debug(f"Full path: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"Image not found: {file_path}")
        return jsonify({'error': 'Image not found'}), 404

    img = cv2.imread(file_path)
    if img is None:
        logger.error(f"cv2.imread failed for path: {file_path}")
        return jsonify({'error': 'Unable to read image'}), 400

    try:
        results = model.predict(source=img, save=False, conf=0.5)
    except Exception as e:
        logger.exception("Model prediction failed")
        return jsonify({'error': str(e)}), 500

    detections = []
    for result in results:
        for box in result.boxes:
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

        img_bgr = result.plot()

        # Resize and pad
        target_width, target_height = 640, 480
        h, w = img_bgr.shape[:2]
        scaling_factor = min(target_width / w, target_height / h)
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)

        top = (target_height - new_size[1]) // 2
        bottom = target_height - new_size[1] - top
        left = (target_width - new_size[0]) // 2
        right = target_width - new_size[0] - left
        img_bgr = cv2.copyMakeBorder(img_bgr, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))

    logger.debug(f"Detections found: {len(detections)}")

    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'detections': detections,
        'image': f'data:image/jpeg;base64,{img_base64}'
    }), 200

@app.route('/')
def serve_index():
    return send_file('visionx.html')

@app.route('/explore')
def serve_explore():
    return send_file('explorepage.html')

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(os.path.join(app.root_path, 'images'), filename)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'images'), 'favicon.ico')

if __name__ == '__main__':
    # Use port 7860 as expected by Hugging Face Spaces
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=port)