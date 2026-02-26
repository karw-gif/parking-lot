# Parking Lot Object Detection System

This project uses the **YOLOv3 object detection model** to detect cars in a parking lot image and attempt to identify occupied parking spaces.

## 🔍 Overview

The system loads a pretrained YOLOv3 model (configuration and class names from `yolov3.cfg` and `coco.names`) and applies it to an input image or video to detect objects — particularly vehicles — in a parking lot scene.

Once detections are made, the output image is saved (e.g., `output.jpeg`) with bounding boxes and labels.

## 📁 Repo Contents

- `ParkingLot.py` – Main Python script for running the YOLOv3 detector on parking images.
- `yolov3.cfg` – YOLOv3 network configuration file.
- `coco.names` – Class names for YOLOv3 detections (80 object classes).
- `output.jpeg` – Example output image with detections.
  
## 🛠 Technologies Used

- Python  
- OpenCV  
- YOLOv3 (Darknet format)  
- Pretrained detection model weights

## 🚀 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/karw-gif/parking-lot.git
   Install required libraries
2.Install the required libraries using -pip install opencv-python numpy
Place an input image in the project directory
Run the detector script -python ParkingLot.py
Review the output image (output.jpeg) with detected cars.
