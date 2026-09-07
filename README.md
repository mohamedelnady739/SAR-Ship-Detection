# 🚢 SAR Ship Detection

AI-powered ship detection from **Synthetic Aperture Radar (SAR)** satellite imagery using **YOLO11s**.

## 📌 Project Overview

This project detects ships in SAR satellite images using a custom-trained YOLO11s object detection model.

The model was trained on the **SARscope** dataset with one class:

* 🚢 Ship

## 🧠 Model

* Model: **YOLO11s**
* Task: **Object Detection**
* Classes: **1**
* Image Size: **640 × 640**
* Training Epochs: **30**

## 📊 Test Results

| Metric    |     Score |
| --------- | --------: |
| Precision | **90.0%** |
| Recall    | **81.7%** |
| mAP@50    | **89.7%** |
| mAP@50-95 | **60.3%** |

## 📁 Dataset

SARscope Dataset:

* Training images: **4,717**
* Validation images: **1,346**
* Test images: **672**
* Total images: **6,735**

## 🚀 Streamlit Application

The application allows users to:

* Upload SAR satellite images
* Detect ships
* Count detected ships
* Display confidence scores
* Display bounding boxes
* View detection details

**Google Drive:**
https://drive.google.com/drive/folders/15EzroX5PJJQjvJOYGLOZk9S-vCIX4aYa?usp=sharing


## 🛠️ Technologies

* Python
* YOLO11
* Ultralytics
* OpenCV
* Streamlit
* PyTorch
* SAR Satellite Imagery

## 🎯 Future Improvements

* Ship tracking with unique IDs
* Video-based ship detection
* Advanced ship counting
* Real-time detection
* Deployment on cloud platforms

AI / Machine Learning / Deep Learning / Computer Vision
