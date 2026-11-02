"""CLI entry point for training and evaluating all EMI models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.model_training import ModelTrainer, prepare_data
from src.mlflow_tracking import MLflowTracker
from src.diagnostics import generate_diagnostics


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(project_root / "data" / "emi_prediction_dataset.csv"))
    parser.add_argument("--output", default=str(project_root / "artifacts"))
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow URI or local directory. Defaults to project-relative mlruns.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Python executable: %s", sys.executable)
    logging.info("Project root: %s", project_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_data(args.data)
    trainer = ModelTrainer()
    fitted, classification, regression = trainer.train_and_evaluate(prepared)
    if classification.empty or regression.empty:
        raise RuntimeError("No validation metrics were produced.")
    classification_model_names = set(classification["model"].tolist())
    for model_name, model in fitted.items():
        artifact = model
        if model_name in classification_model_names:
            artifact = {
                "model": model,
                "label_encoder": trainer.classification_label_encoder,
            }
        joblib.dump(artifact, output / f"{model_name}.joblib")
    classification.to_csv(output / "classification_metrics.csv", index=False)
    regression.to_csv(output / "regression_metrics.csv", index=False)
    test_classification, test_regression = trainer.evaluate_test(prepared, fitted)
    test_classification.to_csv(output / "test_classification_metrics.csv", index=False)
    test_regression.to_csv(output / "test_regression_metrics.csv", index=False)
    test_summary = pd.concat(
        [
            test_classification.assign(task="classification"),
            test_regression.assign(task="regression"),
        ],
        ignore_index=True,
    )
    test_summary.to_json(output / "test_metrics.json", orient="records", indent=2)
    prepared_summary = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "rows": [len(prepared.X_train), len(prepared.X_validation), len(prepared.X_test)],
        "features_before_encoding": [len(prepared.selected_features)] * 3,
    })
    prepared_summary.to_csv(output / "dataset_summary.csv", index=False)
    best_classification = classification.sort_values(
        ["f1_macro", "high_risk_recall", "eligible_recall", "roc_auc_ovr"], ascending=False
    ).iloc[0]
    best_regression = regression.sort_values(["rmse", "mae"], ascending=True).iloc[0]
    joblib.dump(
        {
            "model": fitted[best_classification["model"]],
            "label_encoder": trainer.classification_label_encoder,
        },
        output / "best_classification.joblib",
    )
    joblib.dump(fitted[best_regression["model"]], output / "best_regression.joblib")
    joblib.dump(
        trainer.classification_label_encoder,
        output / "classification_label_encoder.joblib",
    )
    generate_diagnostics(
        prepared,
        output,
        fitted,
        trainer.classification_label_encoder,
    )
    best_classification_predictions = fitted[best_classification["model"]].predict(
        prepared.X_validation
    )
    best_classification_predictions = trainer.classification_label_encoder.inverse_transform(
        best_classification_predictions.astype(int)
    )
    ConfusionMatrixDisplay.from_predictions(
        prepared.y_validation_classification,
        best_classification_predictions,
        labels=trainer.classification_label_encoder.classes_,
        xticks_rotation="vertical",
    )
    plt.tight_layout()
    plt.savefig(output / "best_classification_confusion_matrix.png", dpi=150)
    plt.close()
    best_test_classification = test_classification.sort_values(
        ["f1_macro", "high_risk_recall", "eligible_recall", "roc_auc_ovr"], ascending=False
    ).iloc[0]
    best_test_regression = test_regression.sort_values(["rmse", "mae"]).iloc[0]
    if float(best_test_classification["high_risk_recall"]) < 0.70:
        logging.warning(
            "High_Risk recall is %.3f on the untouched test set; review thresholding "
            "or cost-sensitive training before production use.",
            best_test_classification["high_risk_recall"],
        )
    (output / "selection_summary.txt").write_text(
        f"Best classification: {best_classification['model']}\n"
        f"Best regression: {best_regression['model']}\n"
        f"Validation classification accuracy: {best_classification['accuracy']:.6f}\n"
        f"Test classification accuracy: {best_test_classification['accuracy']:.6f}\n"
        f"Validation regression RMSE: {best_regression['rmse']:.6f}\n"
        f"Test regression RMSE: {best_test_regression['rmse']:.6f}\n"
        f"Dropped training-only features: {prepared.dropped_features}\n",
        encoding="utf-8",
    )
    MLflowTracker.save_summary(classification, regression, output / "metrics.json")
    print("Classification validation metrics:\n", classification.to_string(index=False))
    print("Regression validation metrics:\n", regression.to_string(index=False))
    try:
        tracker = MLflowTracker(tracking_uri=args.tracking_uri, project_root=project_root)
    except RuntimeError as exc:
        logging.warning("%s Training outputs were saved without MLflow logging.", exc)
        return
    for row in classification.to_dict("records"):
        model_name = row["model"]
        tracker.log_run(row["model"], fitted[row["model"]], {}, row,
                        [output / "classification_metrics.csv", output / "dataset_summary.csv",
                         output / "test_classification_metrics.csv",
                         output / "classification_label_encoder.joblib",
                         output / "best_classification_confusion_matrix.png",
                         output / f"{model_name}_test_confusion_matrix.png",
                         output / f"{model_name}_test_roc.png",
                         output / f"{model_name}_test_classification_report.json"],
                        f"EMIPredict-Classification-{row['model']}")
    for row in regression.to_dict("records"):
        model_name = row["model"]
        tracker.log_run(row["model"], fitted[row["model"]], {}, row,
                        [output / "regression_metrics.csv", output / "dataset_summary.csv",
                         output / "test_regression_metrics.csv",
                         output / f"{model_name}_test_predictions.png",
                         output / f"{model_name}_test_residuals.png",
                         output / f"{model_name}_feature_importance.png",
                         output / f"{model_name}_feature_importance.csv"],
                        f"EMIPredict-Regression-{row['model']}")
    tracker.log_run(
        "best-classification",
        fitted[best_classification["model"]],
        {"selection_metric": "f1_weighted"},
        best_classification.to_dict(),
        [output / "best_classification_confusion_matrix.png", output / "selection_summary.txt"],
        "EMIPredict-Best-Classification",
    )
    tracker.log_run(
        "best-regression",
        fitted[best_regression["model"]],
        {"selection_metric": "rmse"},
        best_regression.to_dict(),
        [output / "selection_summary.txt"],
        "EMIPredict-Best-Regression",
    )


if __name__ == "__main__":
    main()
