
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HeartGuard-AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heartguard_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# =========================================================
# 3. FEATURES
# =========================================================

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# =========================================================
# 4. FIXED HIGH-CONTRAST CSS
# =========================================================

st.markdown("""
<style>

/* ===== APP BACKGROUND ===== */
.stApp {
    background: #F4F7FB !important;
}

/* Main content */
[data-testid="stMain"] {
    background-color: #F4F7FB !important;
}

[data-testid="stMain"] p,
[data-testid="stMain"] li,
[data-testid="stMain"] label {
    color: #24364B !important;
}

/* Headings */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #102A43 !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: #12263A !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] small {
    color: #FFFFFF !important;
}

/* ===== INFO / WARNING / SUCCESS / ERROR ALERTS ===== */

/* Info */
[data-testid="stAlert"][data-baseweb="notification"] {
    background-color: #E6F4F7 !important;
    border: 1px solid #9BD2DC !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #173B50 !important;
}

/* ===== INPUT LABELS ===== */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: #24364B !important;
}

/* ===== NUMBER INPUT ===== */
[data-testid="stNumberInput"] {
    background-color: transparent !important;
}

[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    -webkit-text-fill-color: #172B4D !important;
    border: 1px solid #CAD5E2 !important;
    border-radius: 8px !important;
}

[data-testid="stNumberInput"] button {
    background-color: #E9EFF5 !important;
    color: #172B4D !important;
}

/* ===== SELECT BOX ===== */
[data-testid="stSelectbox"] {
    color: #172B4D !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #CAD5E2 !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #172B4D !important;
}

/* Dropdown popup */
[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="popover"] * {
    color: #172B4D !important;
}

[data-baseweb="option"] {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
}

[data-baseweb="option"]:hover {
    background-color: #E8F5F5 !important;
}

/* ===== FORMS ===== */
[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE5EE !important;
    border-radius: 14px !important;
    padding: 24px !important;
}

/* ===== METRIC CARDS ===== */
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE5EE !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

[data-testid="stMetricLabel"] * {
    color: #52667A !important;
}

[data-testid="stMetricValue"] * {
    color: #102A43 !important;
}

/* ===== BUTTONS ===== */
[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    background-color: #0F8B8D !important;
    color: #FFFFFF !important;
    border: 1px solid #0F8B8D !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}

[data-testid="stButton"] button *,
[data-testid="stFormSubmitButton"] button *,
[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
}

[data-testid="stButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
    background-color: #0B7072 !important;
    border-color: #0B7072 !important;
}

/* ===== FILE UPLOADER FIX ===== */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px solid #C9D7E5 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Uploader label */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] label span {
    color: #172B4D !important;
}

/* Uploader dropzone (the black panel in your screenshot) */
[data-testid="stFileUploaderDropzone"] {
    background-color: #F8FAFC !important;
    border: 2px dashed #94A3B8 !important;
    border-radius: 10px !important;
}

/* Text inside dropzone */
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #24364B !important;
}

/* Upload button */
[data-testid="stFileUploaderDropzone"] button {
    background-color: #E6F0F5 !important;
    color: #173B50 !important;
    border: 1px solid #AFC3D3 !important;
}

/* Upload button text */
[data-testid="stFileUploaderDropzone"] button * {
    color: #173B50 !important;
}

/* File size information */
[data-testid="stFileUploaderDropzone"] small {
    color: #52667A !important;
}

/* ===== EXPANDERS ===== */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE5EE !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #172B4D !important;
}

/* ===== TABLES ===== */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE5EE !important;
    border-radius: 10px !important;
}

/* ===== PROGRESS BAR ===== */
[data-testid="stProgressBar"] div[role="progressbar"] {
    background-color: #DCE5EE !important;
}

/* ===== DIVIDERS ===== */
hr {
    border-color: #DCE5EE !important;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    color: #64748B !important;
    padding: 15px 0;
    font-size: 14px;
}

.footer p {
    color: #64748B !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def predict_patient(input_df):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    return (
        prediction,
        float(probabilities[1]),
        float(probabilities[0])
    )


def display_prediction_result(prediction, disease_probability):

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Disease Predicted")
        st.write(
            "The model predicts the presence of heart disease "
            "based on the entered features."
        )
    else:
        st.success("✅ No Disease Predicted")
        st.write(
            "The model predicts the absence of heart disease "
            "based on the entered features."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Disease Probability",
            f"{disease_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Prediction",
            "Disease" if prediction == 1 else "No Disease"
        )

    st.progress(disease_probability)

    st.caption(
        "Probability represents the model output and is not "
        "a validated clinical risk score."
    )


def sample_csv_bytes():

    sample = pd.DataFrame([{
        "age": 45,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 230,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }])

    return sample.to_csv(index=False).encode("utf-8")


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# ❤️ HeartGuard-AI")
    st.markdown("### Heart Disease Prediction")
    st.caption("Machine Learning Application")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Single Prediction",
            "Batch Prediction",
            "About Model"
        ]
    )

    st.divider()

    st.markdown("### Model Details")
    st.write("Algorithm: Logistic Regression")
    st.write("Dataset: Cleveland Heart Disease")
    st.write("Features: 13")
    st.write("Type: Binary Classification")

    st.divider()
    st.caption("HeartGuard-AI")
    st.caption("Educational use only")


# =========================================================
# 7. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("❤️ HeartGuard-AI Dashboard")
    st.subheader("Machine Learning Based Heart Disease Prediction")

    st.info(
        "HeartGuard-AI analyzes selected health-related features "
        "and predicts a binary class using a machine learning model. "
        "This application is for educational purposes only."
    )

    st.subheader("Project Overview")

    c1, c2, c3, c4 = st.columns(4
    )

    c1.metric("Input Features", "13")
    c2.metric("Algorithm", "Logistic Regression")
    c3.metric("Task", "Classification")
    c4.metric("Prediction Modes", "2")

    st.divider()

    st.subheader("Prediction Modes")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 👤 Single Prediction")
        st.write(
            "Enter patient information manually and generate "
            "a prediction for an individual record."
        )

    with c2:
        st.markdown("### 📊 Batch Prediction")
        st.write(
            "Upload a CSV dataset and generate predictions "
            "for multiple records in one operation."
        )

    st.divider()

    st.subheader("Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test Accuracy", "88.52%")
    c2.metric("Test Recall", "92.86%")
    c3.metric("Test F1 Score", "88.14%")
    c4.metric("Test ROC-AUC", "96.65%")

    st.caption(
        "Previously observed experimental results. The test set was "
        "inspected during development; these values are not independent "
        "clinical validation."
    )

    st.divider()

    st.subheader("How It Works")

    st.markdown("""
    **1. Input Data:** Enter patient data or upload a CSV.

    **2. Preprocessing:** Missing values are imputed, numerical
    features are scaled and categorical features are encoded.

    **3. Model Prediction:** Logistic Regression predicts the class.

    **4. Probability Output:** The model returns the probability
    associated with each class.

    **5. Results:** The app displays prediction results and summary.
    """)

    st.warning(
        "This application must not be used for real medical diagnosis."
    )


# =========================================================
# 8. SINGLE PREDICTION
# =========================================================

elif page == "Single Prediction":

    st.title("👤 Single Patient Prediction")
    st.write("Enter all patient information to generate a prediction.")

    st.warning(
        "Educational demonstration only — not for real medical assessment."
    )

    with st.form("single_prediction_form"):

        st.subheader("Personal Information")

        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=45
            )

        with c2:
            sex = st.selectbox(
                "Sex",
                [1, 0],
                format_func=lambda x: "Male" if x == 1 else "Female"
            )

        st.subheader("Cardiovascular Information")

        c1, c2 = st.columns(2)

        with c1:
            cp = st.selectbox(
                "Chest Pain Type",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Typical Angina",
                    2: "Atypical Angina",
                    3: "Non-anginal Pain",
                    4: "Asymptomatic"
                }[x]
            )

            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=50,
                max_value=250,
                value=120
            )

            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=50,
                max_value=600,
                value=200
            )

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with c2:
            restecg = st.selectbox(
                "Resting ECG Result",
                [0, 1, 2],
                format_func=lambda x: {
                    0: "Normal",
                    1: "ST-T Wave Abnormality",
                    2: "Left Ventricular Hypertrophy"
                }[x]
            )

            thalach = st.number_input(
                "Maximum Heart Rate Achieved",
                min_value=50,
                max_value=250,
                value=150
            )

            exang = st.selectbox(
                "Exercise-Induced Angina",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            oldpeak = st.number_input(
                "Oldpeak (ST Depression)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        st.subheader("Additional Features")

        c1, c2, c3 = st.columns(3)

        with c1:
            slope = st.selectbox(
                "Slope of ST Segment",
                [1, 2, 3],
                format_func=lambda x: {
                    1: "Upsloping",
                    2: "Flat",
                    3: "Downsloping"
                }[x]
            )

        with c2:
            ca = st.selectbox(
                "Number of Major Vessels",
                [0, 1, 2, 3]
            )

        with c3:
            thal = st.selectbox(
                "Thal Result",
                [3, 6, 7],
                format_func=lambda x: {
                    3: "Normal",
                    6: "Fixed Defect",
                    7: "Reversible Defect"
                }[x]
            )

        submitted = st.form_submit_button(
            "🔍 Predict",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }], columns=FEATURES)

        try:
            prediction, disease_probability, _ = predict_patient(input_data)

            display_prediction_result(
                prediction,
                disease_probability
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# =========================================================
# 9. BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":

    st.title("📊 Batch Prediction")
    st.write("Upload a CSV file to predict multiple patient records.")

    st.info(
        "Your CSV must contain all 13 required features. "
        "Additional columns are preserved in the results."
    )

    with st.expander("📋 CSV Format and Sample Template"):

        st.write("Required feature columns:")

        st.code(", ".join(FEATURES))

        sample = pd.DataFrame([{
            "age": 45,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 230,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "ca": 0,
            "thal": 3
        }])

        st.dataframe(sample, use_container_width=True)

        st.download_button(
            "⬇️ Download Sample CSV",
            data=sample_csv_bytes(),
            file_name="heartguard_sample.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            if batch_df.empty:
                st.error("The uploaded CSV file is empty.")

            else:

                st.subheader("Dataset Preview")
                st.write(f"Total records: **{len(batch_df)}**")
                st.dataframe(batch_df.head(10), use_container_width=True)

                missing_columns = [
                    col for col in FEATURES
                    if col not in batch_df.columns
                ]

                if missing_columns:

                    st.error(
                        "Missing required columns: "
                        + ", ".join(missing_columns)
                    )

                else:

                    batch_input = batch_df[FEATURES].copy()

                    for col in FEATURES:
                        batch_input[col] = pd.to_numeric(
                            batch_input[col],
                            errors="coerce"
                        )

                    st.success("All required feature columns are present.")

                    if st.button(
                        "🚀 Generate Batch Predictions",
                        use_container_width=True
                    ):

                        try:

                            predictions = model.predict(batch_input)
                            probabilities = model.predict_proba(batch_input)[:, 1]

                            results = batch_df.copy()

                            results["prediction"] = predictions

                            results["prediction_label"] = np.where(
                                predictions == 1,
                                "Disease Predicted",
                                "No Disease Predicted"
                            )

                            results["disease_probability"] = probabilities

                            results["disease_probability_percent"] = (
                                probabilities * 100
                            ).round(2)

                            total = len(results)
                            disease_count = int(np.sum(predictions == 1))
                            no_disease_count = int(np.sum(predictions == 0))
                            average_probability = float(
                                np.mean(probabilities) * 100
                            )

                            st.divider()
                            st.subheader("Batch Summary")

                            c1, c2, c3, c4 = st.columns(4)

                            c1.metric("Total Records", total)
                            c2.metric("Disease Predicted", disease_count)
                            c3.metric("No Disease Predicted", no_disease_count)
                            c4.metric(
                                "Average Probability",
                                f"{average_probability:.2f}%"
                            )

                            st.subheader("Prediction Distribution")

                            chart_data = pd.DataFrame({
                                "Prediction": [
                                    "Disease Predicted",
                                    "No Disease Predicted"
                                ],
                                "Count": [
                                    disease_count,
                                    no_disease_count
                                ]
                            })

                            st.bar_chart(
                                chart_data.set_index("Prediction")
                            )

                            st.subheader("Detailed Results")

                            st.dataframe(
                                results,
                                use_container_width=True
                            )

                            csv_data = results.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                "⬇️ Download Prediction Results",
                                data=csv_data,
                                file_name="heartguard_batch_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                            st.success("Batch prediction completed successfully.")

                            st.warning(
                                "Predictions are model outputs only, "
                                "not medical diagnoses."
                            )

                        except Exception as e:
                            st.error(f"Batch prediction error: {e}")

        except Exception as e:
            st.error(f"CSV reading error: {e}")


# =========================================================
# 10. ABOUT MODEL
# =========================================================

elif page == "About Model":

    st.title("ℹ️ About HeartGuard-AI")

    st.write("""
    HeartGuard-AI is a machine learning project that explores
    heart disease prediction using structured patient data.
    """)

    st.divider()

    st.subheader("Machine Learning Pipeline")

    st.markdown("""
    **Preprocessing**
    - Median imputation for numerical features
    - Most-frequent imputation for categorical features
    - StandardScaler for numerical features
    - OneHotEncoder for categorical features

    **Model**
    - Logistic Regression
    - Binary classification
    """)

    st.divider()

    st.subheader("Input Features")

    feature_info = {
        "age": "Age in years",
        "sex": "Sex category",
        "cp": "Chest pain type",
        "trestbps": "Resting blood pressure",
        "chol": "Serum cholesterol",
        "fbs": "Fasting blood sugar indicator",
        "restecg": "Resting ECG result",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise-induced angina",
        "oldpeak": "ST depression",
        "slope": "Slope of ST segment",
        "ca": "Number of major vessels",
        "thal": "Thal category"
    }

    info_df = pd.DataFrame(
        list(feature_info.items()),
        columns=["Feature", "Description"]
    )

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Dataset")

    st.markdown("""
    **Cleveland Heart Disease Dataset**
    - 303 original records
    - 13 input features
    - Original target values: 0–4
    - Converted to binary:
        - 0 = No disease
        - 1 = Disease present (original target greater than 0)
    """)

    st.divider()

    st.subheader("Limitations")

    st.markdown("""
    - Small dataset
    - Historical dataset
    - Test set inspected during development
    - Probability outputs not clinically calibrated
    - Feature associations do not imply causation
    - No clinical validation
    """)

    st.error(
        "HeartGuard-AI is a student project for educational use only. "
        "Do not use it for medical diagnosis or treatment decisions."
    )

# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.markdown("""
<div class="footer">
    ❤️ HeartGuard-AI | Machine Learning Project | Educational use only
</div>
""", unsafe_allow_html=True)
