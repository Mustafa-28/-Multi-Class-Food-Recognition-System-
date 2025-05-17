# -Multi-Class-Food-Recognition-System-
Food Image Classification System
A PyTorch-based CNN model that classifies food images into three categories: pizza, steak, and sushi.

Key Features
Built from scratch without fine-tuning

Custom data preprocessing pipeline

Handles image classification with a simple, clean interface

Demonstrates core computer vision and deep learning principles

Technical Details
Framework: PyTorch

Model: Custom Convolutional Neural Network (CNN)

Dataset: Curated collection of pizza, steak, and sushi images

Training: Implemented data loading, augmentation, and evaluation

Usage
Clone the repository

Install dependencies (pip install -r requirements.txt)

Run training/evaluation scripts

python
# Example inference  
model = FoodClassifier()  
prediction = model.predict("your_image.jpg")  
Project Structure
├── data/          # Training datasets  
├── models/        # Saved model weights  
├── utils/         # Preprocessing & helper functions  
├── train.py       # Training script  
└── predict.py     # Inference script  
This project serves as a practical implementation of image classification for beginners and showcases PyTorch workflows. Contributions welcome!
