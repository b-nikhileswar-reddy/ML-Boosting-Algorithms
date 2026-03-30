# ⚡ ML Boosting Algorithms: XGBoost & LightGBM

## 📌 Overview
This repository showcases implementations of two powerful boosting algorithms — **XGBoost** and **LightGBM** — applied to structured/tabular datasets.  
Both are widely used in machine learning for tasks like classification, regression, and prediction problems.  

---

## 🚀 Why Boosting?
Boosting is an ensemble technique that builds strong learners by combining multiple weak learners (decision trees).  
It iteratively improves performance by focusing on errors made in previous rounds, making it highly effective for complex datasets.

---

## 📂 Folder Structure
- `XGBoost/` → Implementation of XGBoost algorithm  
- `LightGBM/` → Implementation of LightGBM algorithm  

---

## 📊 XGBoost vs LightGBM

| Feature                | XGBoost                                                                 | LightGBM                                                                 |
|-------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **Tree Growth Strategy**| Level-wise (balanced trees, slower but more stable)                     | Leaf-wise (deeper trees, faster, higher accuracy but risk of overfitting) |
| **Speed**               | Slower compared to LightGBM                                             | Faster due to histogram-based learning                                    |
| **Memory Usage**        | Higher                                                                 | Lower (optimized memory efficiency)                                       |
| **Categorical Features**| Requires one-hot encoding                                               | Native support for categorical features                                   |
| **Best Use Cases**      | Smaller datasets, when stability is preferred                          | Large datasets, when speed and efficiency are critical                    |

---

## 🔧 Requirements
Install dependencies via pip:
```bash
pip install xgboost lightgbm numpy pandas scikit-learn matplotlib seaborn
```
