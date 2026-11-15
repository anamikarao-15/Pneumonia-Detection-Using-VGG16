# Pneumonia Detection Using VGG16

## Project Overview
This project uses Deep Learning and Transfer Learning with the VGG16 model to classify chest X-ray images into two categories:
- NORMAL
- PNEUMONIA

The project also includes a Streamlit web application where users can upload a chest X-ray image and view the model's prediction.

## Dataset
- Dataset: Chest X-Ray Images (Pneumonia)
- Source: Kaggle
- Classes: NORMAL and PNEUMONIA
- Image size: 224 × 224 pixels

## Technologies Used
- Python
- TensorFlow and Keras
- VGG16 (Transfer Learning)
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Google Colab

## Model Architecture
- Pre-trained VGG16 with ImageNet weights
- Global Average Pooling
- Dense layer with 128 neurons and ReLU activation
- Dropout layer (0.5)
- Sigmoid output layer

## Model Performance

**Test Accuracy: 88.62%**

| Metric | NORMAL | PNEUMONIA |
|---|---:|---:|
| Precision | 0.91 | 0.87 |
| Recall | 0.77 | 0.96 |
| F1-Score | 0.84 | 0.91 |

## Web Application
The Streamlit application allows users to:
- Upload chest X-ray images.
- Get a NORMAL or PNEUMONIA prediction.
- View prediction confidence scores.

## Live Demo
[Open Live Application](https://pneumonia-detection-vgg16.streamlit.app/)

## How to Run Locally

1. Clone the repository:
   `git clone https://github.com/anamikarao-15/Pneumonia-Detection-Using-VGG16.git`

2. Install dependencies:
   `pip install -r requirements.txt`

3. Run the application:
   `streamlit run app.py`

## Disclaimer
This project is developed for educational purposes only. It is not a clinically validated diagnostic tool and must not be used as a substitute for professional medical advice.

## Author
**Anamika Rao**  
B.Tech CSE (AI & Data Science)
