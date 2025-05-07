# 🔍 Real-Time Object Detection with YOLOv8 & OpenCV
This project demonstrates real-time object detection using YOLOv8 integrated with OpenCV. Designed to detect custom objects such as fire extinguishers, toolboxes, and oxygen tanks, the solution supports both offline image inference and live webcam detection with GPU acceleration.

## Live Demo Link 
https://youtu.be/_CZmuAoGcgo
https://drive.google.com/file/d/1CRHrsAdgvrFnt1Tlrng6ef3YRtPmlqBh/view?usp=sharing

## 📌 Features
* 🚀 YOLOv8-powered object detection

* 🎥 Live detection using OpenCV and webcam

* 🖼️ Inference on custom test image datasets

* 🧠 Training from scratch on custom data

* 💾 Auto-saving of predicted images and bounding box labels

* 🧼 GPU memory management & OOM error handling

* 📊 Evaluation metrics on test data

## 📁 Directory Structure
```bash
.
├── train.py                 # Training script with custom params
├── infer_images.py         # Static test image inference
├── live_detect.py          # Real-time webcam detection (optional)
├── yolo_params.yaml        # Configuration file (paths, classes, params)
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── runs/                   # YOLOv8 training results
└── predictions/            # Saved predictions and labels
```
## 🛠 Requirements
* Python ≥ 3.8

* Ultralytics YOLOv8

* OpenCV

* PyTorch (with GPU support)

* PyYAML

## Install dependencies:

```bash
pip install ultralytics opencv-python pyyaml torch
```
## 🏋️‍♂️ Training
Update yolo_params.yaml with your custom dataset paths and classes:

```yaml
train: data/train/images
val: data/val/images
test: data/test
nc: 3
names: ['FireExtinguisher', 'ToolBox', 'OxygenTank']
```
Then run:

```bash
python app.py
```
You can modify hyperparameters like epochs, learning rate, batch size directly in the script or via CLI args.


## 🎥 Live Detection with Webcam
Enable live detection using OpenCV:

```bash
python live_detect.py
```
Make sure your webcam is connected and accessible.

## 📊 Evaluation
Validation metrics are generated after inference using the model's .val() method on your test set.

## 📌 Notes
Previous training results are stored in separate runs/detect/train* folders. You can delete old ones if storage is a concern.

Use GPU where available for best performance (device=0 is used by default).

AMP (mixed precision) is enabled by default for faster training.

## 💬 License
This project is licensed under the MIT License. Feel free to use and modify it for personal or commercial projects.
