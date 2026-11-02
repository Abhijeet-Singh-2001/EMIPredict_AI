"""Leakage-safe preprocessing and statistical analysis for EMIPredict AI."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

LOGGER = logging.getLogger(__name__)
MISSING = "missing"
CLASS_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"


@dataclass(frozen=True)
class DataSplits:
    """Raw, cleaned splits created before any learned transformation."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class DataPreprocessor:
    """Load, clean, and split data without fitting transformations on held-out data."""

    def __init__(
        self,
        classification_target: str = CLASS_TARGET,
        regression_target: str = REGRESSION_TARGET,
        random_state: int = 42,
    ) -> None:
        self.classification_target = classification_target
        self.regression_target = regression_target
        self.random_state = random_state

    @staticmethod
    def _clean_column_name(name: object) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
        return cleaned.strip("_")

    def load_and_clean(self, path: str | Path) -> pd.DataFrame:
        """Read a CSV and apply only deterministic, pre-split cleaning."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Dataset not found: {source}")
        try:
            frame = pd.read_csv(source, low_memory=False)
        except (OSError, pd.errors.ParserError) as exc:
            raise ValueError(f"Could not read dataset {source}: {exc}") from exc
        if frame.empty:
            raise ValueError("The dataset is empty.")

        frame.columns = [self._clean_column_name(column) for column in frame.columns]
        if frame.columns.duplicated().any():
            duplicates = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate column names after cleaning: {duplicates}")
        required = {self.classification_target, self.regression_target}
        missing_targets = required.difference(frame.columns)
        if missing_targets:
            raise ValueError(f"Missing required target columns: {sorted(missing_targets)}")

        frame = frame.drop_duplicates().reset_index(drop=True)
        # The requested sentinel is applied to every column before any split.
        frame = frame.replace({r"^\s*$": np.nan}, regex=True).fillna(MISSING)
        LOGGER.info("Loaded %d rows and %d columns from %s", *frame.shape, source)
        return frame

    def split(self, frame: pd.DataFrame) -> DataSplits:
        """Create a stratified 70/15/15 split from the cleaned frame."""
        from sklearn.model_selection import train_test_split

        labels = frame[self.classification_target].astype(str)
        train, holdout = train_test_split(
            frame,
            test_size=0.30,
            random_state=self.random_state,
            stratify=labels,
        )
        holdout_labels = holdout[self.classification_target].astype(str)
        validation, test = train_test_split(
            holdout,
            test_size=0.50,
            random_state=self.random_state,
            stratify=holdout_labels,
        )
        return DataSplits(train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True))

    def feature_target_frames(self, split: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Return raw features and both targets, excluding targets from features."""
        targets = {self.classification_target, self.regression_target}
        features = split.drop(columns=list(targets)).copy()
        return features, split[self.classification_target].copy(), split[self.regression_target].copy()


class StatisticalAnalyzer:
    """Run training-only tests used for transparent feature review."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    @staticmethod
    def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(frame[column], errors="coerce")

    def numeric_regression_tests(self, train: pd.DataFrame, target: str = REGRESSION_TARGET) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        y = self._numeric(train, target)
        for column in train.columns:
            if column == target:
                continue
            x = self._numeric(train, column)
            valid = x.notna() & y.notna()
            if valid.sum() < 3 or x[valid].nunique() < 2:
                continue
            pearson = stats.pearsonr(x[valid], y[valid])
            spearman = stats.spearmanr(x[valid], y[valid])
            rows.append(
                {"feature": column, "pearson_r": pearson.statistic, "pearson_p": pearson.pvalue,
                 "spearman_r": spearman.statistic, "spearman_p": spearman.pvalue}
            )
        return pd.DataFrame(rows).sort_values("spearman_p", ignore_index=True) if rows else pd.DataFrame()

    def categorical_classification_tests(
        self, train: pd.DataFrame, target: str = CLASS_TARGET
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        labels = train[target].astype(str)
        for column in train.columns:
            if column == target or pd.api.types.is_numeric_dtype(train[column]):
                continue
            table = pd.crosstab(train[column].astype(str), labels)
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            chi2, p_value, dof, _ = stats.chi2_contingency(table)
            rows.append({"feature": column, "chi2": chi2, "p_value": p_value, "degrees_of_freedom": dof})
        return pd.DataFrame(rows).sort_values("p_value", ignore_index=True) if rows else pd.DataFrame()

    def numeric_classification_tests(
        self, train: pd.DataFrame, target: str = CLASS_TARGET
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        labels = train[target].astype(str)
        for column in train.columns:
            if column == target:
                continue
            values = self._numeric(train, column)
            groups = [values[labels == group].dropna() for group in labels.unique()]
            groups = [group for group in groups if len(group) >= 2]
            if len(groups) < 2 or sum(group.nunique() > 1 for group in groups) < 2:
                continue
            normal_p = min((stats.normaltest(group).pvalue for group in groups if len(group) >= 8), default=0.0)
            if normal_p >= self.alpha:
                statistic, p_value = stats.f_oneway(*groups)
                test_name = "ANOVA"
            else:
                statistic, p_value = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis"
            rows.append({"feature": column, "test": test_name, "statistic": statistic, "p_value": p_value})
        return pd.DataFrame(rows).sort_values("p_value", ignore_index=True) if rows else pd.DataFrame()

    def vif_report(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate VIF on numeric training data; non-numeric columns are excluded."""
        numeric = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        numeric = numeric.dropna(axis=1, how="all").fillna(numeric.median(numeric_only=True))
        numeric = numeric.loc[:, numeric.nunique(dropna=False) > 1]
        if numeric.shape[1] < 2:
            return pd.DataFrame(columns=["feature", "vif"])
        values = numeric.to_numpy(dtype=float)
        values = np.column_stack([np.ones(len(values)), values])
        vif_values: List[float] = []
        for index in range(1, values.shape[1]):
            response = values[:, index]
            predictors = np.delete(values, index, axis=1)
            coefficients, *_ = np.linalg.lstsq(predictors, response, rcond=None)
            residuals = response - predictors @ coefficients
            total_ss = np.sum((response - response.mean()) ** 2)
            r_squared = 1.0 - (np.sum(residuals**2) / total_ss) if total_ss else 1.0
            vif_values.append(float("inf") if r_squared >= 1.0 else 1.0 / (1.0 - r_squared))
        return pd.DataFrame({"feature": numeric.columns, "vif": vif_values}).sort_values(
            "vif", ascending=False, ignore_index=True
        )

    def report(self, train: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
        """Produce all reports using only the training split."""
        reports = {
            "numeric_vs_regression": self.numeric_regression_tests(train),
            "categorical_vs_classification": self.categorical_classification_tests(train),
            "numeric_vs_classification": self.numeric_classification_tests(train),
            "vif": self.vif_report(train.drop(columns=[CLASS_TARGET, REGRESSION_TARGET], errors="ignore")),
        }
        for name, result in reports.items():
            LOGGER.info("%s: %d rows", name, len(result))
        return reports


class FeatureEngineer:
    """Create domain features and construct a train-fitted preprocessing pipeline."""

    NUMERIC_BASE = (
        "monthly_salary", "monthly_rent", "school_fees", "college_fees", "travel_expenses",
        "groceries_utilities", "other_monthly_expenses", "current_emi_amount",
        "credit_score", "bank_balance", "emergency_fund", "years_of_employment",
        "family_size", "dependents", "requested_amount", "requested_tenure",
    )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add financial ratios while preserving original columns."""
        result = frame.copy()
        numeric = {
            column: pd.to_numeric(
                result[column] if column in result else pd.Series(MISSING, index=result.index),
                errors="coerce",
            ).fillna(0.0)
            for column in self.NUMERIC_BASE
        }
        salary = numeric["monthly_salary"].clip(lower=1.0)
        living_expenses = sum(numeric[column] for column in self.NUMERIC_BASE[1:7])
        total_obligations = living_expenses + numeric["current_emi_amount"]
        result["total_monthly_obligations"] = total_obligations
        result["debt_to_income_ratio"] = total_obligations / salary
        result["expense_to_income_ratio"] = living_expenses / salary
        result["current_emi_to_income_ratio"] = numeric["current_emi_amount"] / salary
        result["disposable_income"] = numeric["monthly_salary"] - total_obligations
        result["available_emi_capacity"] = (salary * 0.40) - total_obligations
        tenure = numeric["requested_tenure"].clip(lower=1.0)
        result["requested_amount_per_tenure"] = numeric["requested_amount"] / tenure
        result["requested_to_annual_income"] = numeric["requested_amount"] / (salary * 12.0)
        result["liquidity_months"] = (numeric["bank_balance"] + numeric["emergency_fund"]) / salary
        result["dependents_per_family_member"] = (
            numeric["dependents"] / numeric["family_size"].clip(lower=1.0)
        )
        result["financial_stability_score"] = (
            numeric["credit_score"].clip(300, 900) / 900
            + np.log1p(numeric["years_of_employment"].clip(lower=0)) / 10
            + np.log1p(numeric["bank_balance"].clip(lower=0)) / 20
        )
        for column in result.columns:
            converted = pd.to_numeric(result[column], errors="coerce")
            if converted.notna().mean() >= 0.95:
                result[column] = converted
            else:
                result[column] = result[column].astype(str)
        return result

    @staticmethod
    def select_features(
        train_features: pd.DataFrame, reports: Mapping[str, pd.DataFrame], vif_threshold: float = 20.0
    ) -> Tuple[List[str], List[str]]:
        """Remove only constant and severely collinear training features.

        Weak univariate p-values are reported but not automatically removed: financial
        predictors can have conditional value after interactions and nonlinear modeling.
        """
        dropped: List[str] = []
        kept = [column for column in train_features.columns if train_features[column].nunique(dropna=False) > 1]
        dropped.extend(column for column in train_features.columns if column not in kept)
        vif = reports.get("vif_engineered", reports.get("vif", pd.DataFrame()))
        protected = {
            "total_monthly_obligations", "debt_to_income_ratio", "expense_to_income_ratio",
            "current_emi_to_income_ratio", "disposable_income", "available_emi_capacity",
            "requested_amount_per_tenure", "requested_to_annual_income", "liquidity_months",
            "dependents_per_family_member", "financial_stability_score",
        }
        if not vif.empty:
            dropped.extend(
                feature for feature in vif.loc[vif["vif"] > vif_threshold, "feature"]
                if feature not in protected
            )
        dropped = sorted(set(dropped))
        return [column for column in kept if column not in dropped], dropped

    def build_pipeline(self, features: pd.DataFrame) -> Pipeline:
        """Build a transformation pipeline; call fit only with training features."""
        numeric = features.select_dtypes(include=np.number).columns.tolist()
        categorical = [column for column in features.columns if column not in numeric]
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", RobustScaler()),
                ]), numeric),
                ("categorical", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), categorical),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline([("preprocessor", preprocessor)])


def build_training_artifacts(
    data_path: str | Path,
) -> Tuple[DataSplits, Dict[str, pd.DataFrame], List[str], List[str], Pipeline]:
    """Convenience entry point used by the notebook."""
    preprocessor = DataPreprocessor()
    cleaned = preprocessor.load_and_clean(data_path)
    splits = preprocessor.split(cleaned)
    analyzer = StatisticalAnalyzer()
    reports = dict(analyzer.report(splits.train))
    engineer = FeatureEngineer()
    train_features, _, _ = preprocessor.feature_target_frames(splits.train)
    train_features = engineer.transform(train_features)
    reports["vif_engineered"] = analyzer.vif_report(train_features)
    selected, dropped = engineer.select_features(train_features, reports)
    pipeline = engineer.build_pipeline(train_features[selected])
    return splits, reports, selected, dropped, pipeline
