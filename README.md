# EMIPredict AI

## Financial risk assessment for responsible EMI decisions

EMIPredict AI is an end-to-end machine-learning project that estimates:

1. **EMI eligibility**: `Eligible`, `High_Risk`, or `Not_Eligible`
2. **Maximum safe monthly EMI**: a continuous INR estimate

The project combines leakage-safe data preparation, domain-driven affordability features, statistical analysis, six benchmark models, MLflow experiment tracking, and a five-page Streamlit application.

> **Important:** This is a decision-support prototype, not an automated lending or adverse-action system. The model output should be reviewed with approved business rules, explainability, fairness checks, and human oversight.

## Why this project matters

EMI decisions should consider more than salary alone. The pipeline combines income, household expenses, existing EMI obligations, liquidity, credit profile, employment stability, and requested-loan details to produce an affordability-oriented assessment.

The business priority is balanced:

- Avoid incorrectly flagging genuinely eligible applicants.
- Still identify High_Risk applicants reliably enough for review.
- Select models using macro F1, High_Risk recall, and Eligible recall rather than majority-class accuracy alone.
- Treat the prediction as a screening signal, not a final credit decision.

## Verified results

The latest untouched-test evaluation used a stratified 70/15/15 train-validation-test split.

### Classification

| Model | Accuracy | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 87.58% | 85.57% | 90.94% |
| Random Forest | 95.28% | 95.03% | 98.45% |
| XGBoost | **97.08%** | **96.78%** | **99.31%** |

### Regression

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | ₹5,207.70 | ₹3,693.09 | 0.5477 |
| Random Forest | ₹1,103.61 | ₹314.84 | 0.9797 |
| XGBoost | **₹1,049.56** | ₹356.80 | **0.9816** |

The project targets of 90% classification accuracy and regression RMSE below ₹2,000 are met. However, the current XGBoost test report shows approximately **50.63% High_Risk recall**. This minority-class limitation is explicitly surfaced rather than hidden by the strong weighted metrics. Cost-sensitive tuning and threshold review are required before production use.

## Architecture

```text
CSV
  ↓
Cleaning and duplicate removal
  ↓
Immediate stratified train/validation/test split
  ↓
Training-only statistical analysis and feature selection
  ↓
Financial feature engineering
  ↓
Training-fitted imputation, scaling, and one-hot encoding
  ↓
Six model pipelines
  ↓
Validation and untouched-test evaluation
  ↓
MLflow tracking and model registry
  ↓
Streamlit risk-assessment application
```

## Repository structure

```text
EMIPredict_AI/
├── app.py                              Streamlit application
├── train.py                            Reproducible training CLI
├── requirements.txt                    Runtime dependencies
├── runtime.txt                         Streamlit Python runtime
├── README.md                           Project documentation
├── .gitattributes                      Git LFS rules
├── .streamlit/config.toml              Streamlit configuration
├── .github/workflows/ci.yml            CI quality checks
├── src/
│   ├── emi_preprocessing.py             Cleaning, splitting, analysis, features
│   ├── model_training.py                Six model families and metrics
│   ├── mlflow_tracking.py               MLflow tracking and registry
│   └── diagnostics.py                   Test plots and per-class reports
├── notebooks/
│   └── 01_data_preprocessing.ipynb      Structured preprocessing walkthrough
├── data/
│   └── emi_prediction_dataset.csv       Anonymized project dataset
└── artifacts/
    ├── best_classification.joblib      Compact selected classifier
    ├── best_regression.joblib          Compact selected regressor
    ├── classification_label_encoder.joblib
    └── metrics and dataset summaries
```

Large Random Forest training artifacts and the local MLflow database are intentionally excluded from the public repository. They remain available locally for reproducibility and comparison.

## Data and leakage prevention

- 404,800 source rows and 27 columns were observed in the supplied CSV.
- Column names are normalized to stable snake case.
- Exact duplicate rows are removed.
- Blank and null values are normalized to the project’s documented `missing` representation.
- The split occurs immediately after initial cleaning and before learned transformations.
- Splitting is stratified by `emi_eligibility` with `random_state=42`.
- Imputers, scalers, encoders, statistical selection, and model fitting use training data only.
- Validation and test partitions are not used to fit preprocessing decisions.

## Financial feature engineering

The pipeline creates interpretable affordability signals:

- `total_monthly_obligations`: rent, education, travel, groceries/utilities, other expenses, and current EMI
- `debt_to_income_ratio`: total obligations relative to monthly salary
- `expense_to_income_ratio`: non-debt household expenses relative to income
- `current_emi_to_income_ratio`: existing EMI burden relative to income
- `disposable_income`: salary remaining after monthly obligations
- `available_emi_capacity`: conservative remaining capacity for a new EMI
- `requested_amount_per_tenure`: requested amount normalized by repayment tenure
- `requested_to_annual_income`: loan request relative to annual income
- `liquidity_months`: bank balance and emergency fund relative to obligations
- `dependents_per_family_member`: household dependency pressure
- `financial_stability_score`: combined credit, liquidity, employment, and affordability signal

These features encode financial relationships that are more useful than isolated raw columns while remaining explainable to loan officers.

## Models

### Classification

- Logistic Regression: interpretable baseline
- Random Forest Classifier: nonlinear ensemble benchmark
- XGBoost Classifier: high-performance boosting model

### Regression

- Linear Regression: interpretable baseline
- Random Forest Regressor: nonlinear ensemble benchmark
- XGBoost Regressor: high-performance boosting model

Classification reports include accuracy, weighted and macro precision/recall/F1, ROC-AUC, High_Risk recall, and Eligible recall. Regression reports include RMSE, MAE, R², and MAPE.

## Run locally

```powershell
git clone https://github.com/Abhijeet-Singh-2001/EMIPredict_AI.git
cd EMIPredict_AI

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Use existing compact artifacts for the application, or retrain everything:
python train.py --data data\emi_prediction_dataset.csv --output artifacts

streamlit run app.py
```

Generate diagnostics from persisted models:

```powershell
python -m src.diagnostics
```

Start the local MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///C:/absolute/path/to/EMIPredict_AI/mlflow.db
```

## Streamlit pages

1. **Home & Overview**: project purpose, architecture, and business use cases
2. **Real-time Risk Assessment**: customer inputs and eligibility/EMI predictions
3. **EDA & Visualizations**: distributions, correlations, and risk segmentation
4. **Model Performance & MLflow**: validation metrics and experiment summary
5. **Administrative Data Management**: session-scoped CSV review and download

## MLflow and artifact policy

MLflow tracks parameters, metrics, model versions, dataset summaries, and diagnostic artifacts. The local SQLite backend is suitable for development. A production deployment should use a managed or hosted MLflow backend rather than a local database inside Streamlit Cloud.

Only the compact selected models are intended for Git LFS deployment. Do not commit credentials, `mlflow.db`, `.venv`, raw confidential data, or multi-gigabyte comparison artifacts.

## CI/CD and deployment

GitHub Actions validates Python syntax and notebook JSON structure. Streamlit Cloud can deploy `app.py` directly from the public GitHub repository using `requirements.txt`.

Before sharing a public URL:

1. Confirm the dataset is anonymized and approved for publication.
2. Confirm the compact model artifacts are present through Git LFS.
3. Deploy `app.py` from the `main` branch.
4. Test all five pages and edge-case inputs.
5. Document the verified public URL.

## Limitations and responsible-use checklist

- High_Risk recall requires cost-sensitive tuning before production decisions.
- Probability calibration and threshold selection have not yet been approved by a lending stakeholder.
- Fairness and subgroup performance audits should be completed before deployment.
- Model drift monitoring and periodic retraining are not yet implemented.
- Pickle/joblib artifacts must only be loaded from trusted, controlled sources.
- A human reviewer must remain in the decision loop.

## Skills demonstrated

Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, statistical testing, feature engineering, leakage prevention, MLflow, model registry, Streamlit, GitHub Actions, Git LFS, documentation, and responsible ML communication.
