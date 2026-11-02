import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from ucimlrepo import fetch_ucirepo

st.set_page_config(
    page_title="SVM Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f8fa;
}
h1, h2, h3 {
    color: #123b46;
}
div.stButton > button {
    background-color: #087f8c;
    color: white;
    border-radius: 10px;
    width: 100%;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load("svm_heart_model.joblib")
    columns = joblib.load("svm_feature_columns.joblib")
    return model, list(columns)


@st.cache_data
def load_dataset():
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Convert the original target into two classes:
    # 0 = no disease, 1 = disease indicator
    target = y.iloc[:, 0]
    target = pd.to_numeric(target, errors="coerce")
    valid = target.notna()

    X = X.loc[valid].copy()
    y_binary = (target.loc[valid] > 0).astype(int)

    return X, y_binary


st.title("❤️ Heart Disease Prediction")
st.caption("Machine Learning Dashboard | Support Vector Machine (SVM)")
st.write(
    "Enter feature values to get a prediction from your trained model. "
    "Explore the dataset charts below."
)

try:
    model, feature_columns = load_model()
except Exception as error:
    st.error(
        "Could not load the saved model. Make sure these files are in "
        "the same folder as app.py:\n"
        "- svm_heart_model.joblib\n"
        "- svm_feature_columns.joblib"
    )
    st.code(str(error))
    st.stop()


# ----------------------------
# SIDEBAR: PATIENT INPUT
# ----------------------------

st.sidebar.header("Enter Feature Values")
st.sidebar.caption("Enter the values for all model features.")

defaults = {
    "age": 50,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 230,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3
}

labels = {
    "age": "Age (years)",
    "sex": "Sex (0 or 1)",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar (0 or 1)",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina (0 or 1)",
    "oldpeak": "Oldpeak",
    "slope": "ST Slope",
    "ca": "Major Vessels (CA)",
    "thal": "Thal"
}

values = {}

with st.sidebar.form("prediction_form"):
    for column in feature_columns:
        default = defaults.get(column, 0.0)

        if column in ["sex", "fbs", "exang"]:
            values[column] = st.selectbox(
                labels.get(column, column),
                options=[0, 1],
                index=int(default)
            )
        elif column in ["cp", "restecg", "slope", "ca", "thal"]:
            values[column] = st.number_input(
                labels.get(column, column),
                value=int(default),
                step=1
            )
        elif column == "oldpeak":
            values[column] = st.number_input(
                labels.get(column, column),
                value=float(default),
                step=0.1
            )
        else:
            values[column] = st.number_input(
                labels.get(column, column),
                value=float(default),
                step=1.0
            )

    predict_clicked = st.form_submit_button("Predict Result")


# ----------------------------
# PREDICTION
# ----------------------------

if predict_clicked:
    try:
        input_data = pd.DataFrame(
            [[values[column] for column in feature_columns]],
            columns=feature_columns
        )

        prediction = int(model.predict(input_data)[0])

        st.subheader("Prediction Result")

        if prediction == 0:
            st.success(
                "Model prediction: Class 0 — no disease indicator "
                "according to the training target."
            )
        else:
            st.warning(
                "Model prediction: Class 1 — disease indicator "
                "according to the training target."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.write("Model output probabilities:")

            probability_df = pd.DataFrame({
                "Class": [
                    str(label)
                    for label in model.classes_
                ],
                "Probability (%)": probabilities * 100
            })

            fig = px.bar(
                probability_df,
                x="Class",
                y="Probability (%)",
                color="Class",
                title="Prediction Probabilities",
                text_auto=".1f"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "This saved model does not provide predict_proba(). "
                "The predicted class is shown above."
            )

        with st.expander("View submitted feature values"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error("Prediction failed. Check the feature names and data types.")
        st.code(str(error))


# ----------------------------
# DATASET CHARTS
# ----------------------------

st.divider()
st.header("Dataset Analysis")

try:
    X, y = load_dataset()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(X))

    with col2:
        st.metric("Number of Features", X.shape[1])

    with col3:
        st.metric("Target Classes", y.nunique())

    # Class distribution
    st.subheader("1. Class Distribution")

    class_counts = (
        y.value_counts()
        .reindex([0, 1], fill_value=0)
        .rename_axis("Class")
        .reset_index(name="Count")
    )

    fig_class = px.bar(
        class_counts,
        x="Class",
        y="Count",
        color="Class",
        text="Count",
        title="Heart Disease Target Distribution"
    )
    st.plotly_chart(fig_class, use_container_width=True)

    # Feature distributions
    st.subheader("2. Feature Distribution")

    available_features = [
        column for column in X.columns
        if pd.api.types.is_numeric_dtype(X[column])
    ]

    selected_feature = st.selectbox(
        "Choose a feature",
        available_features
    )

    fig_hist = px.histogram(
        X,
        x=selected_feature,
        nbins=25,
        title=f"Distribution of {selected_feature}",
        labels={selected_feature: selected_feature}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation heatmap
    st.subheader("3. Feature Correlation")

    numeric_X = X.select_dtypes(include=np.number)
    correlation = numeric_X.corr()

    fig_corr = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Dataset preview
    st.subheader("4. Dataset Preview")

    preview = X.copy()
    preview["target"] = y
    st.dataframe(preview.head(20), use_container_width=True)

except Exception as error:
    st.warning(
        "The saved model can still be used if it loaded successfully, "
        "but the online dataset charts could not be loaded. "
        "Check your internet connection and ucimlrepo installation."
    )
    st.code(str(error))


st.divider()
st.caption(
    "Educational project only. Predictions are not medical diagnoses. "
    "The charts describe the dataset and do not establish medical risk."
)