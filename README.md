# Bank Customer Churn — ML Project

Predict customer churn and identify the key drivers behind exit behavior in a European retail bank, using a 10,000-customer dataset.

---

## Objective

Build an end-to-end machine learning pipeline that:

1. Predicts which customers are likely to churn.
2. Explains *why* they are at risk, in business-readable terms.
3. Surfaces the findings in a dashboard that supports retention decisions.

---

## Dataset

- **File:** `data/raw/Bank_churn.csv`
- **Rows / columns:** 10,000 × 13
- **Target:** `Exited` (1 = churned, 0 = retained)
- **Class balance:** ~20% churn / ~80% retained (imbalanced)

Features: `CreditScore`, `Geography` (France / Germany / Spain), `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`. `CustomerId` and `Surname` are dropped as non-predictive identifiers.

---

## Project Structure

```
bank-customer-churn-project/
├── data/
│   ├── raw/              # Original Kaggle CSV
│   └── processed/        # Cleaned / engineered feature tables
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── outputs/
│   ├── figures/          # All saved charts (EDA, ROC, SHAP, etc.)
│   └── models/           # Serialized model + scaler artifacts
├── src/                  # (planned) reusable Python modules
├── dashboard/            # (planned) Tableau workbook
├── INSTRUCTIONS.md       # Step-by-step build guide
├── project.md            # Original project brief
└── README.md
```

---

## Methodology

| Stage | What happens |
|---|---|
| 1. Data understanding | Schema check, dtype audit, missing/duplicate scan, target balance. |
| 2. EDA | Univariate distributions, target-conditioned views, correlation heatmap. |
| 3. Feature engineering | Encode categoricals (label / one-hot), derive ratios (`BalancePerProduct`, `BalanceSalaryRatio`, `TenureAgeRatio`), `IsZeroBalance` flag, age bands. |
| 4. Train/test split | 80 / 20, stratified on the target, `random_state=42`. |
| 5. Modeling | Three models trained on the same split: Logistic Regression, Random Forest, XGBoost. Class imbalance handled via `class_weight='balanced'` / `scale_pos_weight`. |
| 6. Evaluation | Accuracy, precision, recall, F1, ROC-AUC. Confusion matrices and ROC curves saved. Decision threshold tuned on the precision-recall curve. |
| 7. Explainability | RF feature importance + SHAP global (bar / beeswarm) and per-customer (waterfall) plots. |
| 8. Segmentation | KMeans clustering on behavioral features; clusters profiled and labeled with business names; per-customer scores joined with risk bands. |
| 9. Business insights | Translate findings into retention recommendations (planned). |
| 10. Dashboard | [Tableau Public dashboard](https://public.tableau.com/app/profile/nguyen.gwen/viz/Bank-Churn-Project/Summary) — KPI tiles, segment scatter, churn breakdowns by geography / age band / products / activity, risk band distribution, SHAP top drivers. |

---

## Results

**Winning model: Random Forest**, evaluated on the held-out 20% test set.

| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.25** | **0.472** | 0.779 | **0.588** | **0.86** |
| XGBoost | 0.35 | 0.451 | **0.794** | 0.575 | 0.85 |
| Logistic Regression | 0.50 | — | — | — | 0.79 |

The decision threshold was lowered from the default 0.5 to 0.25 to prioritize **recall** — in a retention context, missing a churner (a lost customer lifetime) is significantly more expensive than a false alarm (a low-cost retention nudge). At this threshold, the model catches ~78% of churners with ~47% precision, surfacing ~15% of customers as elevated risk.

### Top churn drivers (by mean |SHAP|)

1. **Age** — older customers churn more; effect rises sharply after 45.
2. **NumOfProducts** — non-monotonic; customers with 3–4 products churn at near-100% rates.
3. **IsActiveMember** — inactive customers churn at roughly 2× the rate of active ones.
4. **Geography (Germany)** — German customers churn at ~32% vs ~16% in France and Spain.
5. **Balance** — counter-intuitively, *higher* balances correlate with churn (likely affluent customers shopping for better service).

---

## How to Reproduce

### 1. Clone and enter the project

```bash
git clone https://github.com/trngphng1311/bank-customer-churn-project.git
cd bank-customer-churn-project
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyter xgboost shap
```

> **macOS only:** XGBoost requires the OpenMP runtime: `brew install libomp`.

### 3. Launch Jupyter and run the notebooks in order

```bash
jupyter notebook
```

Open and run top-to-bottom:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_modeling.ipynb`

Generated artifacts will appear under `outputs/figures/` and `outputs/models/`, and processed data under `data/processed/`.

---

## Tech Stack

- **Language:** Python 3.12
- **Core libs:** pandas, NumPy
- **Modeling:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Plotting:** matplotlib, seaborn
- **Notebook environment:** Jupyter
- **Dashboard:** [Tableau Public — Bank Churn Project](https://public.tableau.com/app/profile/nguyen.gwen/viz/Bank-Churn-Project/Summary)

---

## Notes / Caveats

- The scaler is fit on the training split only and applied to the test split, to avoid data leakage. The fitted scaler is saved as `outputs/models/scaler.joblib`.
- Logistic Regression uses the scaled feature matrix; tree-based models use the unscaled matrix (trees are scale-invariant).
- The threshold (`0.25`) is part of the deployed system. Loading the model without the threshold would silently revert to scikit-learn's default of 0.5.

---

## Roadmap

- [x] **Stage 8 — Customer segmentation** (KMeans on behavioral features, cluster profiling).
- [x] **Stage 10 — Tableau dashboard** — [live on Tableau Public](https://public.tableau.com/app/profile/nguyen.gwen/viz/Bank-Churn-Project/Summary).
- [ ] **Stage 9 — Business insights writeup** (retention recommendations tied to drivers).
- [ ] Refactor reusable code from notebooks into `src/` modules.
- [ ] Add unit tests for the feature pipeline.

---

## License

This project is for educational and portfolio purposes. The Bank Churn dataset originates from a public Kaggle source and is included for reproducibility.
