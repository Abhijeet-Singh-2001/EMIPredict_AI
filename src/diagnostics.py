"""Generate model diagnostics from persisted pipelines and untouched test data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .model_training import PreparedData, prepare_data

CLASSIFIER_NAMES = (
    "logistic_regression",
    "random_forest_classifier",
    "xgboost_classifier",
)
REGRESSOR_NAMES = (
    "linear_regression",
    "random_forest_regressor",
    "xgboost_regressor",
)


def _feature_names(model: object) -> np.ndarray:
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def generate_diagnostics(
    prepared: PreparedData,
    artifacts_dir: str | Path,
    models: Mapping[str, object],
    label_encoder: object,
) -> None:
    """Create classification, regression, and importance artifacts."""
    output = Path(artifacts_dir)
    output.mkdir(parents=True, exist_ok=True)
    classes = label_encoder.classes_
    encoded_test = label_encoder.transform(prepared.y_test_classification)

    for name in CLASSIFIER_NAMES:
        model = models[name]
        predictions = model.predict(prepared.X_test).astype(int)
        decoded = label_encoder.inverse_transform(predictions)
        ConfusionMatrixDisplay.from_predictions(
            prepared.y_test_classification,
            decoded,
            labels=classes,
            xticks_rotation="vertical",
        )
        plt.tight_layout()
        plt.savefig(output / f"{name}_test_confusion_matrix.png", dpi=150)
        plt.close()

        probabilities = model.predict_proba(prepared.X_test)
        binary_actual = label_binarize(encoded_test, classes=np.arange(len(classes)))
        figure, axis = plt.subplots(figsize=(7, 6))
        for index, class_name in enumerate(classes):
            false_positive, true_positive, _ = roc_curve(
                binary_actual[:, index], probabilities[:, index]
            )
            axis.plot(
                false_positive,
                true_positive,
                label=f"{class_name} (AUC={auc(false_positive, true_positive):.3f})",
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"{name} ROC curves")
        axis.legend(loc="lower right")
        figure.tight_layout()
        figure.savefig(output / f"{name}_test_roc.png", dpi=150)
        plt.close(figure)

        report = classification_report(
            prepared.y_test_classification,
            decoded,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
        (output / f"{name}_test_classification_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    for name in REGRESSOR_NAMES:
        model = models[name]
        actual = prepared.y_test_regression.to_numpy(dtype=float)
        predictions = np.asarray(model.predict(prepared.X_test), dtype=float)
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(actual, predictions, s=4, alpha=0.15)
        limits = [min(actual.min(), predictions.min()), max(actual.max(), predictions.max())]
        axis.plot(limits, limits, linestyle="--", color="red")
        axis.set(xlabel="Actual maximum EMI", ylabel="Predicted maximum EMI", title=f"{name} predictions")
        figure.tight_layout()
        figure.savefig(output / f"{name}_test_predictions.png", dpi=150)
        plt.close(figure)

        residuals = actual - predictions
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(predictions, residuals, s=4, alpha=0.15)
        axis.axhline(0, linestyle="--", color="red")
        axis.set(xlabel="Predicted maximum EMI", ylabel="Residual", title=f"{name} residuals")
        figure.tight_layout()
        figure.savefig(output / f"{name}_test_residuals.png", dpi=150)
        plt.close(figure)

        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": _feature_names(model),
                "importance": estimator.feature_importances_,
            }).sort_values("importance", ascending=False)
            importance.to_csv(output / f"{name}_feature_importance.csv", index=False)
            top = importance.head(20).sort_values("importance")
            top.plot.barh(x="feature", y="importance", legend=False, figsize=(8, 7), title=f"{name} feature importance")
            plt.tight_layout()
            plt.savefig(output / f"{name}_feature_importance.png", dpi=150)
            plt.close()


def load_saved_models(artifacts_dir: str | Path) -> tuple[dict[str, object], object]:
    """Load persisted models and the shared classification encoder."""
    output = Path(artifacts_dir)
    models: dict[str, object] = {}
    for name in (*CLASSIFIER_NAMES, *REGRESSOR_NAMES):
        artifact = joblib.load(output / f"{name}.joblib")
        models[name] = artifact["model"] if isinstance(artifact, dict) else artifact
    return models, joblib.load(output / "classification_label_encoder.joblib")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    prepared_data = prepare_data(root / "data" / "emi_prediction_dataset.csv")
    saved_models, encoder = load_saved_models(root / "artifacts")
    generate_diagnostics(prepared_data, root / "artifacts", saved_models, encoder)
