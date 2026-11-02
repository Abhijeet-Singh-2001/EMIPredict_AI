"""MLflow tracking and model registry integration with explicit optional dependency errors."""

from __future__ import annotations

import json
import logging
import numbers
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

import pandas as pd

LOGGER = logging.getLogger(__name__)


def normalize_tracking_uri(
    tracking_uri: str | Path | None = None,
    project_root: Path | None = None,
) -> str:
    """Return a supported MLflow URI for a local path or explicit URI.

    Filesystem paths and URIs are different representations. In particular,
    ``%20`` is valid in a URI but must be decoded before constructing a
    Windows ``Path``. Local stores use SQLite because MLflow's filesystem
    backend is in maintenance mode in current MLflow releases.
    """
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    value = str(tracking_uri).strip() if tracking_uri is not None else ""
    if not value:
        return f"sqlite:///{(root / 'mlflow.db').as_posix()}"

    parsed = urlparse(value)
    is_windows_drive_path = len(value) >= 2 and value[1] == ":"
    if parsed.scheme and parsed.scheme != "file" and not is_windows_drive_path:
        if parsed.scheme == "sqlite":
            return value
        return value

    if parsed.scheme == "file" and not is_windows_drive_path:
        local_path = Path(unquote(parsed.path.lstrip("/")))
        if parsed.netloc:
            local_path = Path(f"\\\\{parsed.netloc}\\{local_path}")
    else:
        local_path = Path(unquote(value)).expanduser()
    local_path = local_path.resolve()
    database_path = local_path / "mlflow.db" if local_path.suffix.lower() != ".db" else local_path
    return f"sqlite:///{database_path.as_posix()}"


class MLflowTracker:
    """Track model parameters, metrics, artifacts, and registered model versions."""

    def __init__(
        self,
        experiment_name: str = "EMIPredict-AI",
        tracking_uri: str | Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow is required for experiment tracking. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc
        self.mlflow = mlflow
        normalized_uri = normalize_tracking_uri(tracking_uri, project_root)
        if normalized_uri.startswith("sqlite:///"):
            database_path = Path(normalized_uri.removeprefix("sqlite:///"))
            database_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(normalized_uri)
        LOGGER.info("Using MLflow tracking URI: %s", normalized_uri)
        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        run_name: str,
        model: object,
        params: Mapping[str, object],
        metrics: Mapping[str, float],
        artifacts: list[str | Path] | None = None,
        registered_model_name: str | None = None,
    ) -> str:
        """Log one completed model run and optionally register the model."""
        with self.mlflow.start_run(run_name=run_name) as run:
            model_params = model.get_params() if hasattr(model, "get_params") else {}
            combined_params = {**model_params, **params}
            self.mlflow.log_params({key: str(value) for key, value in combined_params.items()})
            numeric_metrics = {
                key: float(value) for key, value in metrics.items()
                if isinstance(value, numbers.Real) and not isinstance(value, bool)
            }
            self.mlflow.log_metrics(numeric_metrics)
            if artifacts:
                for artifact in artifacts:
                    artifact_path = Path(artifact)
                    if artifact_path.is_file():
                        self.mlflow.log_artifact(str(artifact_path))
            if registered_model_name:
                self.mlflow.sklearn.log_model(
                    model,
                    name="model",
                    serialization_format="pickle",
                    registered_model_name=registered_model_name,
                )
            LOGGER.info("Logged MLflow run %s (%s)", run.info.run_id, run_name)
            return run.info.run_id

    @staticmethod
    def save_summary(
        classification_metrics: pd.DataFrame,
        regression_metrics: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """Persist a portable metric summary for the application dashboard."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classification": classification_metrics.to_dict(orient="records"),
            "regression": regression_metrics.to_dict(orient="records"),
        }
        destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return destination
