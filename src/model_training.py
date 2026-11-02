"""Reproducible model training and evaluation for EMI risk prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from .emi_preprocessing import (
    CLASS_TARGET,
    REGRESSION_TARGET,
    DataPreprocessor,
    DataSplits,
    FeatureEngineer,
    StatisticalAnalyzer,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Prepared feature matrices and targets for all dataset partitions."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train_classification: pd.Series
    y_validation_classification: pd.Series
    y_test_classification: pd.Series
    y_train_regression: pd.Series
    y_validation_regression: pd.Series
    y_test_regression: pd.Series
    selected_features: list[str]
    dropped_features: list[str]


def prepare_data(data_path: str | Path, random_state: int = 42) -> PreparedData:
    """Build raw split-specific features while fitting selection on training only."""
    preprocessor = DataPreprocessor(random_state=random_state)
    splits = preprocessor.split(preprocessor.load_and_clean(data_path))
    engineer = FeatureEngineer()
    analyzer = StatisticalAnalyzer()

    raw_train, y_train_cls, y_train_reg = preprocessor.feature_target_frames(splits.train)
    raw_val, y_val_cls, y_val_reg = preprocessor.feature_target_frames(splits.validation)
    raw_test, y_test_cls, y_test_reg = preprocessor.feature_target_frames(splits.test)
    train = engineer.transform(raw_train)
    validation = engineer.transform(raw_val)
    test = engineer.transform(raw_test)
    reports = analyzer.report(splits.train)
    reports["vif_engineered"] = analyzer.vif_report(train)
    selected, dropped = engineer.select_features(train, reports)
    LOGGER.info("Selected %d features; dropped %d training-only features", len(selected), len(dropped))
    return PreparedData(
        train[selected], validation[selected], test[selected],
        y_train_cls.astype(str), y_val_cls.astype(str), y_test_cls.astype(str),
        pd.to_numeric(y_train_reg, errors="coerce"),
        pd.to_numeric(y_val_reg, errors="coerce"),
        pd.to_numeric(y_test_reg, errors="coerce"),
        selected, dropped,
    )


class ModelTrainer:
    """Train model families using a shared, train-fitted preprocessing pipeline."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.engineer = FeatureEngineer()
        self.classification_label_encoder: LabelEncoder | None = None

    def _xgboost(self, task: str):
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is required for XGBoost models. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc
        common = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                  "subsample": 0.85, "colsample_bytree": 0.85, "random_state": self.random_state,
                  "n_jobs": -1}
        if task == "classification":
            return XGBClassifier(**common, objective="multi:softprob", eval_metric="mlogloss")
        return XGBRegressor(**common, objective="reg:squarederror", eval_metric="rmse")

    def models(self) -> Mapping[str, Mapping[str, object]]:
        """Return the required three models for each task."""
        classification: Dict[str, object] = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                tol=1e-4,
                solver="newton-cg",
                random_state=self.random_state,
            ),
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=250, random_state=self.random_state, n_jobs=-1, class_weight="balanced"
            ),
            "xgboost_classifier": self._xgboost("classification"),
        }
        regression: Dict[str, object] = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=250, random_state=self.random_state, n_jobs=-1
            ),
            "xgboost_regressor": self._xgboost("regression"),
        }
        return {"classification": classification, "regression": regression}

    def _pipeline(self, features: pd.DataFrame, estimator: object):
        from sklearn.pipeline import Pipeline

        return Pipeline([
            ("preprocessor", self.engineer.build_pipeline(features).named_steps["preprocessor"]),
            ("model", estimator),
        ])

    def train_and_evaluate(self, prepared: PreparedData) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
        """Fit all required models and return fitted pipelines plus validation metrics."""
        fitted: Dict[str, object] = {}
        classification_rows = []
        regression_rows = []
        self.classification_label_encoder = LabelEncoder()
        y_train_classification = pd.Series(
            self.classification_label_encoder.fit_transform(prepared.y_train_classification),
            index=prepared.y_train_classification.index,
            name=CLASS_TARGET,
        )
        try:
            y_validation_classification = pd.Series(
                self.classification_label_encoder.transform(prepared.y_validation_classification),
                index=prepared.y_validation_classification.index,
                name=CLASS_TARGET,
            )
            self.classification_label_encoder.transform(prepared.y_test_classification)
        except ValueError as exc:
            raise ValueError(
                "Validation or test classification labels contain a class absent from the "
                "training split; cannot evaluate safely."
            ) from exc

        for task, candidates in self.models().items():
            for name, estimator in candidates.items():
                target = y_train_classification if task == "classification" else prepared.y_train_regression
                validation_target = (
                    y_validation_classification if task == "classification"
                    else prepared.y_validation_regression
                )
                valid = ~pd.isna(target)
                pipe = self._pipeline(prepared.X_train, clone(estimator))
                pipe.fit(prepared.X_train.loc[valid], target.loc[valid])
                predictions = pipe.predict(prepared.X_validation)
                fitted[name] = pipe
                if task == "classification":
                    probabilities = pipe.predict_proba(prepared.X_validation)
                    classes = pipe.named_steps["model"].classes_
                    decoded_predictions = self.classification_label_encoder.inverse_transform(
                        np.asarray(predictions, dtype=int)
                    )
                    decoded_validation = self.classification_label_encoder.inverse_transform(
                        np.asarray(validation_target, dtype=int)
                    )
                    row = {
                        "model": name, "accuracy": accuracy_score(decoded_validation, decoded_predictions),
                        "precision_weighted": precision_score(decoded_validation, decoded_predictions, average="weighted", zero_division=0),
                        "recall_weighted": recall_score(decoded_validation, decoded_predictions, average="weighted", zero_division=0),
                        "f1_weighted": f1_score(decoded_validation, decoded_predictions, average="weighted", zero_division=0),
                        "precision_macro": precision_score(decoded_validation, decoded_predictions, average="macro", zero_division=0),
                        "recall_macro": recall_score(decoded_validation, decoded_predictions, average="macro", zero_division=0),
                        "f1_macro": f1_score(decoded_validation, decoded_predictions, average="macro", zero_division=0),
                        "high_risk_recall": recall_score(
                            decoded_validation, decoded_predictions, labels=["High_Risk"],
                            average="macro", zero_division=0,
                        ),
                        "eligible_recall": recall_score(
                            decoded_validation, decoded_predictions, labels=["Eligible"],
                            average="macro", zero_division=0,
                        ),
                        "roc_auc_ovr": roc_auc_score(validation_target, probabilities, labels=classes, multi_class="ovr"),
                    }
                    if hasattr(pipe.named_steps["model"], "n_iter_"):
                        row["optimizer_iterations"] = int(np.max(pipe.named_steps["model"].n_iter_))
                    classification_rows.append(row)
                else:
                    actual = validation_target.to_numpy(dtype=float)
                    pred = np.asarray(predictions, dtype=float)
                    row = {
                        "model": name, "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
                        "mae": mean_absolute_error(actual, pred), "r2": r2_score(actual, pred),
                        "mape": float(np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100),
                    }
                    regression_rows.append(row)
                LOGGER.info("%s/%s validation metrics: %s", task, name, row)
        return fitted, pd.DataFrame(classification_rows), pd.DataFrame(regression_rows)

    def evaluate_test(
        self,
        prepared: PreparedData,
        fitted: Mapping[str, object],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate fitted models once on the untouched test split."""
        if self.classification_label_encoder is None:
            raise RuntimeError("Classification label encoder is not initialized.")
        encoded_test = pd.Series(
            self.classification_label_encoder.transform(prepared.y_test_classification),
            index=prepared.y_test_classification.index,
        )
        classification_rows = []
        regression_rows = []
        for name, model in fitted.items():
            predictions = model.predict(prepared.X_test)
            if name in {"logistic_regression", "random_forest_classifier", "xgboost_classifier"}:
                probabilities = model.predict_proba(prepared.X_test)
                classes = model.named_steps["model"].classes_
                decoded_predictions = self.classification_label_encoder.inverse_transform(
                    np.asarray(predictions, dtype=int)
                )
                decoded_actual = prepared.y_test_classification.to_numpy()
                classification_rows.append({
                    "model": name,
                    "accuracy": accuracy_score(decoded_actual, decoded_predictions),
                    "precision_weighted": precision_score(decoded_actual, decoded_predictions, average="weighted", zero_division=0),
                    "recall_weighted": recall_score(decoded_actual, decoded_predictions, average="weighted", zero_division=0),
                    "f1_weighted": f1_score(decoded_actual, decoded_predictions, average="weighted", zero_division=0),
                    "precision_macro": precision_score(decoded_actual, decoded_predictions, average="macro", zero_division=0),
                    "recall_macro": recall_score(decoded_actual, decoded_predictions, average="macro", zero_division=0),
                    "f1_macro": f1_score(decoded_actual, decoded_predictions, average="macro", zero_division=0),
                    "high_risk_recall": recall_score(
                        decoded_actual, decoded_predictions, labels=["High_Risk"],
                        average="macro", zero_division=0,
                    ),
                    "eligible_recall": recall_score(
                        decoded_actual, decoded_predictions, labels=["Eligible"],
                        average="macro", zero_division=0,
                    ),
                    "roc_auc_ovr": roc_auc_score(encoded_test, probabilities, labels=classes, multi_class="ovr"),
                })
            else:
                actual = prepared.y_test_regression.to_numpy(dtype=float)
                pred = np.asarray(predictions, dtype=float)
                regression_rows.append({
                    "model": name,
                    "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
                    "mae": mean_absolute_error(actual, pred),
                    "r2": r2_score(actual, pred),
                    "mape": float(np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100),
                })
        return pd.DataFrame(classification_rows), pd.DataFrame(regression_rows)

    @staticmethod
    def classification_artifact(y_true: pd.Series, y_pred: np.ndarray, output: Path) -> Path:
        """Save a confusion matrix plot for experiment tracking."""
        output.parent.mkdir(parents=True, exist_ok=True)
        matrix = confusion_matrix(y_true, y_pred)
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.imshow(matrix, cmap="Blues")
        axis.set_title("Eligibility confusion matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, int(value), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return output
