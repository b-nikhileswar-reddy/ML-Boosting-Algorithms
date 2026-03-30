# 🏥 Medical Cost Prediction with LightGBM

## 📌 Overview
This folder contains an implementation of **Light Gradient Boosting Machine (LightGBM)** applied to **medical cost prediction**.  
The goal is to estimate healthcare expenses based on patient attributes (age, BMI, smoking status, region, etc.), enabling better resource allocation and risk assessment in clinical and insurance settings.

---

## 🚀 Why LightGBM?
LightGBM is a gradient boosting framework developed by Microsoft, optimized for speed and accuracy. It is particularly effective for **structured/tabular healthcare datasets** because:
- ⚡ Fast training with large datasets  
- 🌱 Leaf-wise tree growth for higher accuracy  
- 🧩 Native support for categorical features  
- 🎯 Strong performance in regression tasks like cost prediction  

---

## 📂 Folder Structure
- `Medical_cost_prediction.ipynb`     → Jupyter Notebook with full implementation  
- `medical_cost.csv`                  → Medical insurance dataset (features: age, sex, BMI, children, smoker, region, charges)  
- `medical_cost_prediction_model.pkl` → pickel file of trained model which is used.  

---

## ⚙️ Workflow
1. **Data Preprocessing**
   - Handle missing values
   - Transform Numerical values
   - Encode categorical features (sex, smoker, region)  
   - Train-test split  

2. **Model Training**
   - Implement `LGBMRegressor`  
   - Hyperparameter tuning (learning rate, num leaves, max depth, etc.)  

3. **Evaluation**
   - Metrics: RMSE, R² score  

---

## 📊 Results
- Achieved **low RMSE** compared to baseline linear regression models  
- Demonstrated efficiency: faster training and better accuracy than XGBoost.  

---

## 🔧 Requirements
Install dependencies via pip:
```bash
pip install lightgbm numpy pandas scikit-learn matplotlib seaborn joblib
```

## ▶️ Running the Trained Model and making prediction
```
import pickle
import pandas as pd

# Load the trained model
with open("medical_cost_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# Example input data (replace with real patient info)
sample_data = pd.DataFrame({
    "age": [35],
    "sex": ["male"],
    "bmi": [28.5],
    "children": [2],
    "smoker": ["yes"],
    "region": ["southeast"]
})

# Make prediction
predicted_cost = model.predict(sample_data)
print("Predicted Medical Cost:", predicted_cost[0])
```
