# ❤️ Heart Disease Detection with XGBoost

## 📌 Overview
This project applies **Extreme Gradient Boosting (XGBoost)** to predict the likelihood of heart disease based on patient attributes such as age, sex, cholesterol, blood pressure, and lifestyle factors.  
The goal is to support early detection and preventive healthcare.

---

## 🚀 Why XGBoost?
XGBoost is chosen because it:
- ⚡ Trains quickly and scales well
- 🌱 Includes regularization to reduce overfitting
- 🧩 Handles missing values natively
- 🎯 Delivers strong performance in classification tasks

---

## 📂 Folder Structure
- `Heart_disease_prediction.ipynb` → Jupyter Notebook with full implementation  
- `heart_disease_data.csv` → Dataset containing patient features and target labels  
- `heart_disease_predection_model.pkl` → Trained XGBoost model saved with pickle  
- `results/` → Evaluation outputs  
  - `reports/` → Classification report, confusion matrix  
  - `curves/` → ROC curve, Precision-Recall curve  

---

## ⚙️ Workflow
1. **Data Preprocessing**
   - Handle missing values  
   - Normalize numerical features  
   - Encode categorical variables  
   - Train-test split  

2. **Model Training**
   - Implement `XGBClassifier`  
   - Hyperparameter tuning (learning rate, max depth, n_estimators, etc.)  

3. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC  
   - Visualizations: Confusion Matrix, ROC Curve, PR Curve  

---

## 📊 Results
- Achieved 87% accuracy with 93% precision
- Visualized
  - Classification Report and Confusion matrix
  - ROC-AUC and PR Curves

---

## 🔧 Requirements
Install dependencies via pip:
```bash
pip install xgboost numpy pandas scikit-learn matplotlib seaborn joblib
```

---
## ▶️ Running the Trained Model and making prediction

```
import pickle
import pandas as pd

# Load the trained model
with open("heart_disease_predection_model.pkl", "rb") as file:
    model = pickle.load(file)

# Example input data
sample_data = pd.DataFrame({
    "age": [52],
    "sex": ["male"],
    "cholesterol": [240],
    "blood_pressure": [140],
    "max_heart_rate": [150],
    "exercise_induced_angina": ["yes"]
})

# Make prediction
predicted_class = model.predict(sample_data)
print("Heart Disease Risk:", "Yes" if predicted_class[0] == 1 else "No")
```
