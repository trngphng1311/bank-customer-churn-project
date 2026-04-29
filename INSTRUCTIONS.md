# 🏦 Bank Customer Churn — Step-by-Step Instructions

A self-paced guide to complete the project end-to-end. Follow stages in order; each one builds on the previous.

**Dataset:** `Bank_Churn.csv` — 10,000 rows × 13 columns
**Target:** `Exited` (1 = churned, 0 = retained)
**Features:** CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

---

## 🛠 Stage 0: Environment Setup (Mac)

### 0.1 Project folders
From the project root, create the structure listed in `project.md`:

```bash
cd /Users/nguyenphuong/Documents/code/PROJECT/data/bank-customer-churn
mkdir -p data/raw data/processed notebooks src outputs/figures outputs/models dashboard
mv Bank_Churn.csv data/raw/
```

### 0.2 Python environment
Use a virtual env so dependencies stay isolated:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter xgboost shap
```

Optional but useful: `pip install ipykernel` then `python -m ipykernel install --user --name=bank-churn`.

### 0.3 Launch Jupyter
```bash
jupyter notebook
```
Create the three notebooks under `notebooks/`: `01_eda.ipynb`, `02_feature_engineering.ipynb`, `03_modeling.ipynb`.

---

## 🔹 Stage 1: Data Understanding (notebook `01_eda.ipynb`)

**Goal:** know what you're working with before touching it.

1. Load the CSV: `df = pd.read_csv('../data/raw/Bank_Churn.csv')`
2. Inspect:
   - `df.shape` → confirm (10000, 13)
   - `df.dtypes` → numeric vs object columns
   - `df.head()`, `df.describe()`
   - `df.isnull().sum()` → expect zero missing in this dataset
   - `df.duplicated().sum()` → check duplicates by `CustomerId`
3. Drop columns that leak no signal: `CustomerId`, `Surname` (these are identifiers, not features).
4. Check class balance: `df['Exited'].value_counts(normalize=True)` — you'll see ~20% churn (imbalanced; remember this for Stage 6).

**Checkpoint:** you can describe the dataset in 2 sentences, including class imbalance.

---

## 🔹 Stage 2: Exploratory Data Analysis (`01_eda.ipynb`)

Use **seaborn + matplotlib**. Save every figure to `outputs/figures/` so you can reuse them in the dashboard later.

### 2.1 Univariate
- Histograms: `Age`, `CreditScore`, `Balance`, `EstimatedSalary`, `Tenure`
- Bar charts: `Geography`, `Gender`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`

### 2.2 Target relationship — the heart of EDA
For each feature, compare distribution across `Exited = 0` vs `Exited = 1`:
- **Numeric:** boxplot or KDE split by `Exited`
- **Categorical:** churn rate per category (`groupby(...)['Exited'].mean()`)

Specifically answer:
- Churn rate by **Geography** (Germany usually has the highest — verify)
- Churn rate by **Gender**
- Churn rate by **Age band** (bin into 18–30, 30–45, 45–60, 60+)
- Churn rate by **NumOfProducts** (often non-monotonic — 3 and 4 products show very high churn)
- Churn rate by **IsActiveMember**

### 2.3 Correlation
- `sns.heatmap(df.corr(numeric_only=True), annot=True)` — focus on the `Exited` row.

**Checkpoint:** write 5 bullet points describing what differentiates churners from non-churners. These become your business insights.

---

## 🔹 Stage 3: Feature Engineering (`02_feature_engineering.ipynb`)

1. **Encode categoricals:**
   - `Gender` → 0/1 (label encode, only 2 values)
   - `Geography` → one-hot encode (`pd.get_dummies(..., drop_first=True)`)
2. **Derived features** (these often improve the model):
   - `BalancePerProduct = Balance / NumOfProducts`
   - `BalanceSalaryRatio = Balance / EstimatedSalary`
   - `TenureAgeRatio = Tenure / Age`
   - `IsZeroBalance = (Balance == 0).astype(int)` — large group of zero-balance customers
   - `AgeGroup` bins (then one-hot encode)
3. **Scale numerics** for distance-based / linear models: `StandardScaler` on `CreditScore, Age, Tenure, Balance, EstimatedSalary` and the ratios. Tree models (RF, XGBoost) don't need scaling — keep both versions if you want.
4. Save: `df.to_csv('../data/processed/churn_features.csv', index=False)`

**Checkpoint:** you have a clean feature matrix `X` and target `y`, all numeric, no NaNs.

---

## 🔹 Stage 4: Data Splitting (`03_modeling.ipynb`)

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
**Always pass `stratify=y`** because the classes are imbalanced.

For cross-validation later: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

---

## 🔹 Stage 5: Model Building

Train three models on the same split so they're comparable.

| Model | Library | Notes |
|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression(max_iter=1000, class_weight='balanced')` | Baseline. Use scaled features. |
| Random Forest | `sklearn.ensemble.RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)` | Strong baseline. |
| XGBoost | `xgboost.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, scale_pos_weight=4, eval_metric='logloss')` | Usually wins. `scale_pos_weight ≈ neg/pos ratio` (~4). |

For each model: `.fit(X_train, y_train)` → predict on `X_test`. Save the best model: `joblib.dump(model, '../outputs/models/xgb_churn.joblib')`.

---

## 🔹 Stage 6: Model Evaluation

For each model, compute on the **test set**:
- `accuracy_score`
- `precision_score`, `recall_score`, `f1_score`
- `roc_auc_score` using `predict_proba(...)[:, 1]`
- Confusion matrix (`sklearn.metrics.ConfusionMatrixDisplay`)
- ROC curve (plot all 3 on one chart)

**Optimize for recall on the churn class.** A bank cares more about catching churners (false negatives are expensive) than about false alarms. If recall is low, try:
- Lowering the decision threshold (e.g., 0.35 instead of 0.5)
- Increasing `scale_pos_weight` / `class_weight`
- SMOTE oversampling (`imblearn` package) — optional

Build a comparison table and pick a winner. Document why.

---

## 🔹 Stage 7: Feature Importance & Explainability

1. **Tree-based importance:** `model.feature_importances_` → sorted bar chart.
2. **SHAP** (more trustworthy):
   ```python
   import shap
   explainer = shap.TreeExplainer(xgb_model)
   shap_values = explainer.shap_values(X_test)
   shap.summary_plot(shap_values, X_test)
   ```
   Save the summary plot — it's gold for the dashboard.

**Answer the question "What drives churn?"** in plain English using the top 5 features.

---

## 🔹 Stage 8: Customer Segmentation

1. Use only behavioral features (drop the target).
2. Standardize, then `KMeans(n_clusters=4, random_state=42)`.
3. Use the elbow method (`inertia_`) to validate `k`.
4. Profile each cluster: mean Age, Balance, NumOfProducts, IsActiveMember, churn rate.
5. Label clusters with business names: e.g., "Older inactive Germans (high-risk)", "Young engaged French (loyal)".

Export to CSV for the dashboard: `df_with_clusters.to_csv('../data/processed/churn_segments.csv', index=False)`.

---

## 🔹 Stage 9: Business Insights

Write a short narrative (1 page) covering:
- Headline churn rate and where it concentrates (geography, age, products)
- Top 3 drivers from SHAP
- 2–3 retention recommendations tied to drivers (e.g., "Activity-engagement campaign for German customers aged 45+ holding 1 product")

---

## 🔹 Stage 10: Dashboard / Visualization

### 📌 Recommendation for Mac users

**Power BI Desktop does NOT run on macOS.** Your real options:

| Option | Cost | Mac-friendly? | Best for this project |
|---|---|---|---|
| **Tableau Public (Desktop)** | Free | ✅ Native Mac app | ⭐ **Recommended** — best UX for this dataset, works offline, public link to share |
| Tableau Desktop (paid) | Paid trial | ✅ Native Mac app | Same as Public but private workbooks |
| Power BI Service (web only) | Free tier | ⚠️ Browser only — no Desktop authoring on Mac | Possible, but limited; you upload data and build in browser |
| Looker Studio | Free | ✅ Browser | Good free alternative if you want to skip Tableau |

**My pick: Tableau Public.** Download from `public.tableau.com`. It runs natively on Mac, exports a shareable link great for a portfolio, and handles 10K rows trivially.
**If you specifically want Power BI:** use Power BI Service in the browser — upload the CSV, build visuals in the web canvas. You'll lose some advanced authoring but the basics work.

### Dashboard contents (build the same in either tool)
1. **KPI tiles:** total customers, churn rate %, avg balance of churners, count of high-risk segment
2. **Churn rate by Geography** (bar)
3. **Churn rate by Age band** (bar)
4. **Churn rate by NumOfProducts** (bar — highlight 3 & 4)
5. **Active vs Inactive churn** (donut or stacked bar)
6. **Top features driving churn** (bar from SHAP exported as CSV: feature, mean_abs_shap)
7. **Segment view:** scatter of Age vs Balance, colored by cluster, sized by churn probability
8. **Filters:** Geography, Gender, Age band, IsActiveMember

Export the prepared dataset (`data/processed/churn_segments.csv`) into Tableau as the data source. Save the Tableau workbook (`.twbx`) under `dashboard/`.

---

## 🔹 Stage 11: Final Deliverables Checklist

- [ ] `notebooks/01_eda.ipynb` — clean, with section headers and conclusions
- [ ] `notebooks/02_feature_engineering.ipynb`
- [ ] `notebooks/03_modeling.ipynb` — final model + evaluation + SHAP
- [ ] `outputs/models/<best_model>.joblib`
- [ ] `outputs/figures/` — saved charts (EDA, ROC, confusion matrix, SHAP summary)
- [ ] `data/processed/churn_segments.csv`
- [ ] `dashboard/<your_workbook>.twbx` (Tableau) **or** Power BI Service link
- [ ] `README.md` — project overview, how to run, top insights, dashboard screenshot/link

---

## ⏱ Suggested pacing (working solo, ~2h/day)

| Day | Work |
|---|---|
| 1 | Stages 0–1 |
| 2 | Stage 2 (EDA — the longest) |
| 3 | Stages 3–4 |
| 4 | Stage 5 (training all 3 models) |
| 5 | Stages 6–7 (evaluation + SHAP) |
| 6 | Stage 8 (clustering + profiling) |
| 7 | Stage 10 (dashboard) |
| 8 | Stages 9 + 11 (insights writeup, README) |

---

## 💡 Tips while working solo

- **Commit / snapshot often** — even without git, keep dated copies of notebooks before big changes.
- **Don't tune hyperparameters first.** Get all 3 models running with defaults, then tune the winner.
- **Recall, not accuracy.** A model that predicts "no one churns" gets 80% accuracy and is useless.
- **Sanity-check derived features.** Watch for divide-by-zero (`Balance / NumOfProducts` is fine here since `NumOfProducts ≥ 1`).
- **Write business insights as you go**, not at the end — they're easier to capture while the analysis is fresh.
