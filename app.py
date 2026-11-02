"""Streamlit multi-page EMI eligibility and affordability application."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - exercised by deployment environment
    raise RuntimeError("Install Streamlit with `python -m pip install -r requirements.txt`.") from exc

from src.emi_preprocessing import DataPreprocessor, FeatureEngineer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emi_prediction_dataset.csv"
ARTIFACTS = ROOT / "artifacts"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load cleaned source data for dashboard-only exploration."""
    return DataPreprocessor().load_and_clean(DATA_PATH)


@st.cache_resource
def load_model(name: str) -> Any:
    """Load a persisted training pipeline."""
    path = ARTIFACTS / f"{name}.joblib"
    if not path.is_file():
        return None
    return joblib.load(path)


def _first_available_model(*names: str) -> Any:
    """Return the first persisted model without relying on estimator truthiness."""
    for name in names:
        model = load_model(name)
        if model is not None:
            return model
    return None


def home_page() -> None:
    st.title("EMIPredict AI")
    st.subheader("Intelligent financial risk assessment")
    st.markdown(
        "This application estimates EMI eligibility and a safe maximum monthly EMI using "
        "a leakage-safe, stratified training workflow."
    )
    st.info("Run `python train.py` after installing requirements to populate the model artifacts.")
    st.markdown("### Architecture")
    st.write("Clean and split → engineer financial features → train model families → track with MLflow → serve predictions.")
    st.markdown("### Business use cases")
    st.write("Loan pre-screening, affordability guidance, risk triage, and transparent model monitoring.")


def assessment_page() -> None:
    st.title("Real-time risk assessment")
    st.caption("Inputs are validated before inference. Predictions require trained artifacts in artifacts/.")
    data = load_data()
    model_cls = _first_available_model("best_classification", "random_forest_classifier")
    model_reg = _first_available_model("best_regression", "random_forest_regressor")
    numeric = ["age", "monthly_salary", "years_of_employment", "monthly_rent", "family_size", "dependents",
               "school_fees", "college_fees", "travel_expenses", "groceries_utilities", "other_monthly_expenses",
               "current_emi_amount", "credit_score", "bank_balance", "emergency_fund", "requested_amount", "requested_tenure"]
    categorical = ["gender", "marital_status", "education", "employment_type", "company_type", "house_type",
                   "existing_loans", "emi_scenario"]
    values: dict[str, Any] = {}
    left, right = st.columns(2)
    for index, column in enumerate(numeric):
        default = float(pd.to_numeric(data[column], errors="coerce").median())
        values[column] = (left if index % 2 == 0 else right).number_input(column.replace("_", " ").title(), value=default, min_value=0.0)
    for index, column in enumerate(categorical):
        options = sorted(data[column].astype(str).dropna().unique().tolist())
        values[column] = (left if index % 2 == 0 else right).selectbox(column.replace("_", " ").title(), options)
    if st.button("Assess application", type="primary"):
        if model_cls is None or model_reg is None:
            st.error("No trained models found. Run the training command first.")
            return
        try:
            row = FeatureEngineer().transform(pd.DataFrame([values]))
            classification_model = model_cls["model"] if isinstance(model_cls, dict) else model_cls
            prediction = classification_model.predict(row)
            if isinstance(model_cls, dict) and model_cls.get("label_encoder") is not None:
                prediction = model_cls["label_encoder"].inverse_transform(prediction.astype(int))
            regression_model = model_reg["model"] if isinstance(model_reg, dict) else model_reg
            predicted_emi = float(regression_model.predict(row)[0])
            if not np.isfinite(predicted_emi):
                raise ValueError("The regression model returned a non-finite value.")
            st.success(f"Eligibility: **{prediction[0]}**")
            st.metric("Maximum safe monthly EMI", f"₹{float(np.clip(predicted_emi, 500, 50000)):,.0f}")
        except (ValueError, TypeError, KeyError) as exc:
            logging.exception("Inference failed")
            st.error(f"Could not assess this application: {exc}")


def eda_page() -> None:
    st.title("Exploratory data analysis")
    data = load_data()
    st.metric("Records", f"{len(data):,}")
    st.bar_chart(data["emi_eligibility"].value_counts())
    st.subheader("Maximum EMI distribution")
    st.line_chart(data["max_monthly_emi"].value_counts().sort_index().head(100))
    numeric = data.select_dtypes(include="number")
    st.subheader("Numeric correlation matrix")
    st.dataframe(numeric.corr(numeric_only=True).round(3), use_container_width=True)
    st.subheader("Risk by employment type")
    st.dataframe(pd.crosstab(data["employment_type"], data["emi_eligibility"], normalize="index").round(3))


def performance_page() -> None:
    st.title("Model performance and MLflow dashboard")
    metrics_path = ARTIFACTS / "metrics.json"
    if not metrics_path.is_file():
        st.warning("No metrics found. Run `python train.py` first.")
        return
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Classification validation results")
    st.dataframe(pd.DataFrame(payload["classification"]), use_container_width=True)
    st.subheader("Regression validation results")
    st.dataframe(pd.DataFrame(payload["regression"]), use_container_width=True)
    st.caption("The MLflow tracking UI can be started with `mlflow ui` when MLflow is installed.")


def admin_page() -> None:
    st.title("Administrative data management")
    st.warning("Uploaded records are session-scoped and are not written to the source dataset.")
    data = load_data()
    uploaded = st.file_uploader("Upload CSV for review", type="csv")
    working = pd.read_csv(uploaded) if uploaded else data.copy()
    st.dataframe(working.head(100), use_container_width=True)
    delete_index = st.number_input("Row index to remove from this session", min_value=0, max_value=max(len(working) - 1, 0), value=0)
    if st.button("Remove selected row"):
        st.session_state["admin_data"] = working.drop(index=delete_index, errors="ignore")
        st.success("Row removed from the current session.")
    if "admin_data" in st.session_state:
        st.download_button("Download reviewed CSV", st.session_state["admin_data"].to_csv(index=False), "reviewed_data.csv")


def main() -> None:
    st.set_page_config(page_title="EMIPredict AI", page_icon="💳", layout="wide")
    pages = {
        "Home & Overview": home_page,
        "Real-time Risk Assessment": assessment_page,
        "EDA & Visualizations": eda_page,
        "Model Performance & MLflow": performance_page,
        "Administrative Data Management": admin_page,
    }
    selected = st.sidebar.radio("Navigate", list(pages))
    pages[selected]()


if __name__ == "__main__":
    main()
