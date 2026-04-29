# 🏦 Bank Customer Churn Project (ML Portfolio)

## 🎯 Objective

Predict customer churn and identify key drivers behind customer exit behavior in a European bank dataset.

---

## 📁 Project Structure

```
bank-churn-project/
│── data/
│   ├── raw/
│   ├── processed/
│
│── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│
│── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate.py
│
│── outputs/
│   ├── figures/
│   ├── models/
│
│── dashboard/
│
│── README.md
```

---

## 🔹 Stage 1: Data Understanding

* Load dataset
* Inspect schema and data types
* Identify target variable (Exited)
* Check missing values and duplicates

---

## 🔹 Stage 2: Exploratory Data Analysis (EDA)

### Key Analysis:

* Customer demographics (age, geography)
* Distribution of balance, salary
* Churn rate overall

### Comparative Analysis:

* Churn by country (France, Germany, Spain)
* Churn by age group
* Churn by number of products

### Visualizations:

* Histograms
* Boxplots
* Correlation heatmap

---

## 🔹 Stage 3: Feature Engineering

* Encode categorical variables (Geography, Gender)
* Normalize numerical features (optional)
* Create derived features:

  * Balance per product
  * Activity ratio

---

## 🔹 Stage 4: Data Splitting

* Train/Test split (80/20)
* Optional: Cross-validation

---

## 🔹 Stage 5: Model Building

Train multiple models:

* Logistic Regression
* Random Forest
* XGBoost

---

## 🔹 Stage 6: Model Evaluation

Metrics:

* Accuracy
* Precision / Recall
* F1-score
* ROC-AUC

Focus on:

* Recall (detect churners)

---

## 🔹 Stage 7: Feature Importance & Explainability

* Extract feature importance
* Optional: SHAP values

Answer:

* What drives churn?

---

## 🔹 Stage 8: Customer Segmentation

* Apply clustering (KMeans)
* Identify segments:

  * High-risk customers
  * Loyal customers

---

## 🔹 Stage 9: Business Insights

* Summarize findings
* Example:

  * Inactive users churn more
  * Older customers more likely to churn

---

## 🔹 Stage 10: Dashboard / Visualization

Tools:

* Power BI / Tableau

Include:

* Churn rate by segment
* Key drivers
* Risk distribution

---

## 🔹 Stage 11: Final Deliverables

* Clean notebook
* Trained model
* Dashboard
* README with insights

---

## 🏁 Final Goal

Deliver a complete ML pipeline + business insights that explain churn behavior and suggest retention strategies.
