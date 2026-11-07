SVM – Heart Disease Prediction

<p align="center">
  <strong>A Machine Learning project using Support Vector Machine (SVM) to predict heart disease.</strong>
</p><p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Algorithm-SVM-green" />
  <img src="https://img.shields.io/badge/Project-Classification-purple" />
</p>---

Overview

SVM – Heart Disease Prediction is a Machine Learning project that uses the Support Vector Machine (SVM) algorithm to classify whether a patient has heart disease or not based on medical features.

The project uses the UCI Heart Disease dataset and applies a supervised Machine Learning approach to analyze patient data and make predictions.

SVM works by finding an optimal decision boundary (hyperplane) that separates different classes while maximizing the margin between them.

Features

- Heart disease classification using SVM.
- Uses the UCI Heart Disease dataset.
- Binary classification: Heart Disease / No Heart Disease.
- Data preprocessing and preparation.
- Model training and testing.
- Model evaluation using classification metrics.
- Prediction for new patient records.

Technologies Used

Technology| Purpose
Python| Main programming language
Pandas| Data manipulation
NumPy| Numerical operations
Scikit-learn| Machine Learning implementation
UCI Repository| Dataset source

Dataset

The project uses the Heart Disease Dataset from the UCI Machine Learning Repository.

- Dataset Name: Heart Disease
- Dataset ID: 45
- Number of Features: 13
- Original Target: "num"
- Task: Binary Classification

Target Transformation

The original target column contains values from 0 to 4.

For this project, the target is converted into two classes:

Original Value| New Class
0| No Heart Disease
1, 2, 3, 4| Heart Disease

Input Features

The model uses 13 medical features:

Feature| Description
age| Patient's age
sex| Sex value in the dataset
cp| Chest pain type
trestbps| Resting blood pressure
chol| Serum cholesterol
fbs| Fasting blood sugar
restecg| Resting ECG results
thalach| Maximum heart rate achieved
exang| Exercise-induced angina
oldpeak| ST depression
slope| Slope of peak exercise ST segment
ca| Number of major vessels
thal| Thalassemia-related value

Machine Learning Algorithm

Support Vector Machine (SVM)

SVM is a supervised Machine Learning algorithm used mainly for classification and regression tasks.

It works by finding the best hyperplane that separates data points belonging to different classes.

The main goal of SVM is to maximize the margin between the decision boundary and the closest data points from each class.

These closest data points are called Support Vectors.

How SVM Works

1. Receive the training data and their class labels.
2. Find a decision boundary (hyperplane) that separates the classes.
3. Identify the support vectors closest to the boundary.
4. Maximize the margin between the classes.
5. Use the trained model to classify new data points.

Machine Learning Workflow

1. Load the Heart Disease dataset.
2. Explore and understand the dataset.
3. Separate input features and target.
4. Convert the target into binary classes.
5. Handle missing values and prepare the data.
6. Split the dataset into training and testing sets.
7. Train the SVM classifier.
8. Evaluate the model performance.
9. Use the trained model to make predictions.

Model Evaluation

The model performance can be evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help measure how well the model classifies the test data.

«Add your actual evaluation results here after running the model.»

Project Structure

SVM-Heart-Disease-Prediction/
│
├── ml_model.py
├── ui.py
├── terminal_test.py
├── README.md
│
└── models/

Update the structure to match the actual files in your project.

Installation

Clone the repository:

git clone YOUR_REPOSITORY_LINK

Install the required libraries:

pip install pandas numpy scikit-learn ucimlrepo

How to Run

Run the Machine Learning model:

python ml_model.py

Run the user interface:

python ui.py

Run the terminal testing script:

python terminal_test.py

Use the commands that match your actual project files and interface.

Important Notes

- SVM works by finding an optimal decision boundary.
- Feature scaling is often important for SVM.
- The choice of kernel can affect model performance.
- The regularization parameter (C) controls the trade-off between margin size and classification errors.
- Data preprocessing is necessary before training.

Limitations

- The dataset has a limited number of records.
- SVM performance depends on the selected kernel and parameters.
- Training can become computationally expensive on large datasets.
- The model's performance on this dataset does not guarantee the same performance on real-world medical data.

Future Improvements

- Experiment with different SVM kernels such as Linear, RBF, and Polynomial.
- Tune hyperparameters such as C and gamma.
- Improve data preprocessing.
- Apply feature selection.
- Compare SVM with KNN and Decision Tree algorithms.

Disclaimer

This project is developed for educational and academic purposes only.

It is not a medical diagnostic tool and should not be used to make real medical decisions.

Acknowledgements

- UCI Machine Learning Repository for providing the Heart Disease dataset.
- Scikit-learn for Machine Learning tools and algorithms.
