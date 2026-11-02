import joblib
import pandas as pd

# Load the trained model and feature names
model = joblib.load("svm_heart_model.joblib")
feature_columns = joblib.load("svm_feature_columns.joblib")

print("=" * 40)
print("Heart Disease - SVM Prediction")
print("=" * 40)

print("\nEnter the patient's feature values:")
print("Press Enter without a value to leave it missing.")

patient_data = {}

for feature in feature_columns:
    while True:
        value = input(f"{feature}: ").strip()

        if value == "":
            patient_data[feature] = float("nan")
            break

        try:
            patient_data[feature] = float(value)
            break
        except ValueError:
            print("Please enter a numeric value.")

# Keep the same feature order used during training
patient_df = pd.DataFrame(
    [patient_data],
    columns=feature_columns
)

# Predict using the saved model
prediction = model.predict(patient_df)[0]

print("\n" + "=" * 40)
print("Prediction Result")
print("=" * 40)

if prediction == 0:
    print("Predicted class: 0")
    print("The model predicts the no-disease class.")
else:
    print("Predicted class: 1")
    print("The model predicts the disease class.")

print("\nEducational prediction only, not a medical diagnosis.")