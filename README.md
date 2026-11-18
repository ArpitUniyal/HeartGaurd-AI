# ❤️ HeartGuard-AI

**HeartGuard-AI** is an educational machine-learning project that predicts whether a record is associated with heart disease using patient-related features from the Cleveland Heart Disease dataset. It includes a trained Logistic Regression pipeline and a Streamlit interface for single-record and batch CSV predictions.

> **Disclaimer:** This project is for learning and demonstration purposes only. It is not a medical device and must not be used for diagnosis, treatment, or real-world medical decisions.


## 🚀 Live Demo

**Try HeartGuard-AI here:** [Live Application](https://heartgaurd-ai-ezxqcsqyhtszrurgf2yes6.streamlit.app/)

## Features

- **Single prediction:** Enter the 13 input features manually and receive a model prediction with its disease-class probability.
- **Batch prediction:** Upload a CSV file containing multiple records to generate predictions in bulk.
- **Batch summary:** View record counts, prediction distribution, and average predicted disease probability.
- **Download results:** Export the batch predictions to CSV.
- **Preprocessing pipeline:** Missing-value imputation, numerical scaling, and categorical encoding are packaged with the model.
- **Model exploration:** Notebook-based EDA, model comparison, cross-validation, hyperparameter tuning, threshold analysis, and SHAP explainability.

## Machine Learning Workflow

1. Load and inspect the Cleveland Heart Disease dataset.
2. Explore distributions, categorical values, relationships, correlations, and missing values.
3. Convert the original target (`num`, values 0–4) to a binary target:
   - `0` — no disease (original value is 0)
   - `1` — disease present (original value is greater than 0)
4. Split data into training and test sets using stratification.
5. Preprocess features:
   - Numerical columns: median imputation and `StandardScaler`
   - Categorical columns: most-frequent imputation and `OneHotEncoder`
6. Compare Logistic Regression, Random Forest, and XGBoost using stratified cross-validation.
7. Tune Random Forest and XGBoost with `GridSearchCV`.
8. Evaluate the models and examine classification thresholds.
9. Use SHAP to explore model feature contributions.
10. Train and save the final Logistic Regression preprocessing/model pipeline with Joblib.

## Models Compared

| Model | Cross-validation Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 84.71% | 87.68% | 78.34% | 82.45% | 90.25% |
| Tuned Random Forest | 81.39% | 81.88% | 77.35% | 79.02% | 90.26% |
| Tuned XGBoost | 81.79% | 84.26% | 75.53% | 78.99% | 88.12% |

Logistic Regression was selected as the final candidate based on its overall metric profile, including recall, F1, and ROC-AUC on the evaluated split. The model comparison is exploratory and does not establish clinical superiority.

## Final Candidate Test Results

The reported test-set results for the Logistic Regression candidate:

| Metric | Score |
|---|---:|
| Accuracy | 88.52% |
| Precision | 83.87% |
| Recall | 92.86% |
| F1-score | 88.14% |
| ROC-AUC | 96.65% |

The test set was inspected during development and is not a pristine, untouched holdout. The dataset is small (303 original records), and these results should not be interpreted as independent clinical validation or as evidence of real-world performance.

## Input Features

The application expects these 13 columns in this exact spelling (column order is handled by the app/model pipeline):

| Column | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex category (dataset encoding) |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting ECG category |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression (oldpeak) |
| `slope` | Slope category of the ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thal category |

## Project Structure

```text
HeartGuard-AI/
├── app/                    # Application-related files (if used)
├── data/                    # Dataset files
├── models/
│   └── heartguard_model.joblib
├── notebooks/
│   └── 01_data_understanding.ipynb
├── src/                      # Source code / utilities (if used)
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
├── app.py                     # Streamlit app entry point
├── requirements.txt
└── README.md
```

*Your local folders may contain additional files; adjust this tree to reflect the repository before publishing.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd HeartGuard-AI
```

### 2. Create and activate a virtual environment

**Windows (Command Prompt):**

```bash
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

If Streamlit is not listed in your requirements file, install it with:

```bash
pip install streamlit
```

### 4. Confirm the model file exists

Make sure the trained pipeline is present at:

```text
models/heartguard_model.joblib
```

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) that you can open in your browser.

## Using the App

### Single Prediction

1. Open **Single Prediction** from the sidebar.
2. Enter the patient-feature values in the form.
3. Click **Predict**.
4. Review the predicted class and displayed disease probability.

### Batch Prediction

1. Open **Batch Prediction**.
2. Download the sample CSV template if needed.
3. Prepare a CSV with all 13 required feature columns.
4. Upload the CSV.
5. Click **Generate Batch Predictions**.
6. Review the prediction distribution and result table.
7. Download the generated CSV.

### CSV Example

```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
45,1,2,130,230,0,0,150,0,1.0,2,0,3
58,1,4,140,280,1,1,125,1,2.5,2,1,7
52,0,3,120,220,0,0,160,0,0.5,1,0,3
```

## Technologies Used

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- SHAP
- Joblib
- Streamlit
- Matplotlib

## Explainability

SHAP was used during model analysis to explore global feature importance and individual prediction contributions. SHAP values describe how features influence this model's output; they do not establish causation or medical significance.


## Author

Created as a machine-learning learning project to practice data analysis, preprocessing, model evaluation, explainability, model serialization, and deployment with Streamlit.

