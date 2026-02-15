🩺 Chronic Kidney Disease Prediction using Machine Learning
📌 Project Overview
This project focuses on predicting Chronic Kidney Disease (CKD) using Machine Learning techniques. The system analyzes medical parameters from patient data and classifies whether a patient is likely to have CKD or not.
The model is trained using the Random Forest Classifier, which provides high accuracy and reliable performance for classification problems.
🎯 Objectives
To build a machine learning model for early detection of CKD
To preprocess and clean medical dataset effectively
To evaluate model performance using standard metrics
To deploy the model using a web-based interface
🛠️ Technologies Used
Python
Pandas & NumPy (Data Processing)
Matplotlib & Seaborn (Visualization)
Scikit-learn (Machine Learning)
Django (Web Deployment)
Joblib (Model Saving)
📊 Dataset
The dataset contains medical attributes such as:
Age
Blood Pressure
Specific Gravity
Albumin
Sugar
Serum Creatinine
Hemoglobin
Packed Cell Volume
And other clinical parameters
Target Variable:
0 → No CKD
1 → CKD
⚙️ System Workflow
Load Dataset
Data Preprocessing (Handle missing values, Encode categorical data)
Train-Test Split
Model Training using Random Forest
Model Evaluation (Accuracy, Confusion Matrix, Precision, Recall, F1-Score)
Save Model
Deploy using Django Web Application
📈 Model Evaluation
The model performance is evaluated using:
Accuracy Score
Confusion Matrix
Precision
Recall
F1-Score
Cross-Validation
The model achieved high accuracy in predicting CKD cases.
