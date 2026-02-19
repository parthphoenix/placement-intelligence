# 🎓 Placement & Salary Intelligence Dashboard

A production-ready, end-to-end Machine Learning system that predicts student **placement status** (classification) and **expected salary** (regression) based on academic performance, skills, internships, and certifications.

---

## 📋 Business Problem Statement

Campus placement is a critical milestone for students and universities alike. This system uses historical student data to:

1. **Predict placement outcomes** – helping career counselors identify at-risk students early.
2. **Estimate expected salary** – providing realistic salary benchmarks for placed students.

The dashboard provides actionable insights through interactive visualizations, model interpretability (SHAP), and a deployable REST API.

---

## 📊 Dataset Overview

| Feature | Description |
|---|---|
| Student_ID | Unique identifier |
| Gender | Male / Female |
| Age | Student age |
| Degree | Engineering, CS, Business, Arts, Data Science |
| CGPA | Cumulative GPA (0–4.0) |
| Internships_Count | Number of internships |
| Projects_Count | Number of projects |
| Certifications_Count | Number of certifications |
| Technical_Skills_Score_100 | Technical skills (0–100) |
| Communication_Skills_Score_100 | Communication skills (0–100) |
| Aptitude_Test_Score_100 | Aptitude test score (0–100) |
| Placement_Offer | Yes / No (target – classification) |
| Salary_Offered_USD | Offered salary in USD (target – regression) |

**600 records** · No missing values · Balanced class distribution

---

## 🔧 ML Pipeline

```
Raw Data → Cleaning → Feature Engineering → Train/Test Split (80/20)
         → Preprocessing Pipeline (ColumnTransformer)
         → GridSearchCV (5-fold CV) → Model Selection → SHAP Analysis
```

**Key Design Decisions:**
- Stratified split for classification to preserve class balance
- `ColumnTransformer` with `StandardScaler` (numeric) + `OneHotEncoder` (categorical)
- Preprocessing fitted only on training data (no data leakage)
- `random_state=42` throughout for reproducibility

---

## ⚙️ Feature Engineering

| Feature | Formula |
|---|---|
| `total_skills_count` | (Tech + Comm + Aptitude) / 100 |
| `internship_binary` | 1 if internships > 0 else 0 |
| `certification_count` | Same as Certifications_Count |
| `academic_index` | CGPA × Aptitude / 100 |
| `skill_intensity_score` | Tech × Comm / 100 |

---

## 📈 Model Comparison

### Classification (Placement Prediction)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| **Gradient Boosting** | — | — | — | — | — |

*Best model selected by ROC-AUC. Run training to populate metrics.*

### Regression (Salary Prediction)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Ridge | — | — | — |
| Random Forest Regressor | — | — | — |
| **Gradient Boosting Regressor** | — | — | — |

*Best model selected by R². Run training to populate metrics.*

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd placement-intelligence
pip install -r requirements.txt
```

### 2. Run EDA

```bash
python -m src.eda
```

### 3. Train models

```bash
python -m src.train_classification
python -m src.train_regression
```

### 4. Generate SHAP analysis

```bash
python -m src.shap_analysis
```

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

### 6. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 🔌 API Usage

### POST /predict_placement

```bash
curl -X POST http://localhost:8000/predict_placement \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Age": 22,
    "Degree": "Engineering",
    "CGPA": 3.5,
    "Internships_Count": 2,
    "Projects_Count": 5,
    "Certifications_Count": 3,
    "Technical_Skills_Score_100": 80,
    "Communication_Skills_Score_100": 70,
    "Aptitude_Test_Score_100": 75
  }'
```

**Response:**
```json
{
  "placement_prediction": "Placed",
  "placement_probability": 0.7342
}
```

### POST /predict_salary

```bash
curl -X POST http://localhost:8000/predict_salary \
  -H "Content-Type: application/json" \
  -d '{ ... same input ... }'
```

**Response:**
```json
{
  "predicted_salary": 12450.50
}
```

---

## 🐳 Docker Deployment

### Build image

```bash
docker build -t placement-intelligence .
```

### Run container

```bash
docker run -p 8000:8000 placement-intelligence
```

### Deploy to Render

1. Push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### Deploy to Railway

1. Push this repo to GitHub
2. Create a new project on [Railway](https://railway.app)
3. Connect your GitHub repo
4. Railway auto-detects the Dockerfile

---

## 📁 Project Structure

```
placement-intelligence/
│
├── data/
│   ├── raw/
│   │   └── Student_Placement_Skills_2025.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classification_modeling.ipynb
│   ├── 04_regression_modeling.ipynb
│   └── 05_model_interpretation.ipynb
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── evaluate.py
│   ├── eda.py
│   └── shap_analysis.py
│
├── models/
│   ├── classification_model.pkl
│   └── regression_model.pkl
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── dashboard/
│   └── app.py
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🏆 Resume-Ready Summary

> Built an end-to-end Machine Learning pipeline for student placement prediction and salary estimation. Engineered 5 derived features and compared 7 models using GridSearchCV with 5-fold stratified cross-validation. Deployed as a FastAPI REST API with Pydantic validation and a 3-page Streamlit dashboard featuring interactive predictions and SHAP interpretability. Containerized with Docker for cloud deployment.

---

## 📄 License

MIT License. Built for educational and professional portfolio purposes.
