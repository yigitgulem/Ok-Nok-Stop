Project Name: Hand Gesture Recognition and Classification System

https://github.com/yigitgulem/Ok-Nok-Stop/assets/139170950/a7ca846a-b18d-4d16-9b7d-2ac5ed2207e8

This GitHub repository contains files related to a system that detects and classifies hand gestures using a camera. The system captures hand gestures in real-time using OpenCV and MediaPipe libraries and classifies these gestures into different categories.

Files and Folders:
data.py

Description: This script captures hand gestures from the camera and records the coordinates of these gestures into a CSV file named 'hands_data_nok.csv'.
Technologies Used: OpenCV, MediaPipe
hands_data_nok.csv, hands_data_ok.csv, hands_data_stop.csv

Description: CSV files containing coordinates and labels of hand gestures. These data are used for training machine learning models.
knn_model.joblib, lr_model.joblib, rf_model.joblib, svm_model.joblib

Description: Models trained using different machine learning algorithms (KNN, Logistic Regression, Random Forest, SVM). These models are used to classify hand gestures.
ok_nok_stop.py

Description: A Python script that detects and classifies hand gestures in real-time. It uses the 'svm_model.joblib' model.
Technologies Used: OpenCV, MediaPipe, joblib
ok_nok_stop_knn.ipynb

Description: A Jupyter Notebook file that demonstrates the classification of hand gestures using the KNN model.
Installation and Usage:
Install the required libraries (OpenCV, MediaPipe, joblib, numpy, pandas, scikit-learn, etc.).
Run the data.py script to collect hand gesture data.
Load the trained models using the joblib.load function for use.
Perform real-time classification with the ok_nok_stop.py script
