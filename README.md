# 🏠 California Housing Price Prediction Pipeline

An end-to-end Machine Learning pipeline and interactive web application built with Python, Scikit-Learn, and Streamlit to predict median house values in California districts based on demographic, structural, and geographic features.

---

## 📌 Project Overview
This project fulfills an end-to-end machine learning task:
1. **Exploratory Data Analysis (EDA):** Automated generation of correlation heatmaps and feature distribution plots.
2. **Preprocessing Pipeline:** Feature scaling (StandardScaler) and missing-value handling (SimpleImputer) packaged inside Scikit-Learn's ColumnTransformer.
3. **Model Training:** RandomForestRegressor estimator trained and serialized with joblib compression (models/housing_pipeline.joblib).
4. **Web UI Deployment:** Interactive Streamlit web interface (app.py) allowing users to input neighborhood parameters and receive real-time house price predictions.

---

## 📁 Repository Structure

housing-ml-pipeline/
│
├── models/
│   └── housing_pipeline.joblib  # Serialized Scikit-Learn ML pipeline
│
├── src/
│   ├── eda.py                   # Exploratory Data Analysis & visual generation
│   └── train.py                 # Pipeline construction, training & model export
│
├── app.py                       # Streamlit web user interface
├── requirements.txt             # Project Python dependencies
└── README.md                    # Project documentation

---

## 🛠️ Tech Stack & Requirements
* Language: Python 3.x
* Core Libraries: pandas, numpy, scikit-learn, joblib, matplotlib, seaborn, streamlit

---

## 🚀 How to Run Locally

### 1. Clone the repository
git clone https://github.com/Vorlax899/housing-ml-pipeline.git
cd housing-ml-pipeline

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run Exploratory Data Analysis
Generates summary statistics and exports visualization charts (eda_correlation.png, eda_target_dist.png):
python src/eda.py

### 4. Train & Save the Machine Learning Pipeline
Trains the model, outputs performance metrics (MAE, RMSE, R²), and saves the compressed model pipeline:
python src/train.py

### 5. Launch the Streamlit Web App
streamlit run app.py
Open http://localhost:8501 in your browser to interact with the application.

---

## 📊 Model Evaluation
The pipeline processes 8 feature dimensions (e.g., Median Income, House Age, Latitude/Longitude) using a Random Forest Regressor, delivering reliable predictions evaluated against standard regression metrics:
* MAE: Average dollar deviation from ground truth
* RMSE: Standard deviation of residuals
* R² Score: Captures proportion of variance in housing values
