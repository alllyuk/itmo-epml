"""Model evaluation stage."""

import json
import logging
from pathlib import Path
import yaml

import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params() -> dict:
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def evaluate_model(cfg: DictConfig) -> dict:
    """
    Evaluate trained model and generate reports.

    Steps:
    1. Load model and validation data
    2. Make predictions
    3. Calculate comprehensive metrics
    4. Generate plots data
    5. Log to MLflow
    """
    logger.info("Starting model evaluation...")

    # Create directories
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Load model and data
    model = joblib.load("models/model.joblib")
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    metrics = {
        "eval_mse": float(mse),
        "eval_rmse": float(np.sqrt(mse)),
        "eval_mae": float(mean_absolute_error(y_val, y_pred)),
        "eval_r2": float(r2_score(y_val, y_pred)),
        "eval_mape": float(np.mean(np.abs((y_val - y_pred) / y_val)) * 100),
    }

    # Additional statistics
    residuals = y_val - y_pred
    metrics["residual_mean"] = float(np.mean(residuals))
    metrics["residual_std"] = float(np.std(residuals))

    logger.info(f"Evaluation metrics: {metrics}")

    # Generate plots data
    predictions_df = pd.DataFrame({
        "actual": y_val,
        "predicted": y_pred
    })
    predictions_df.to_csv("reports/figures/predictions_vs_actual.csv", index=False)

    residuals_df = pd.DataFrame({
        "predicted": y_pred,
        "residual": residuals
    })
    residuals_df.to_csv("reports/figures/residuals.csv", index=False)

    # Save metrics
    with open("reports/metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Log to MLflow (connect to existing run if available)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    # Try to find the training run and log evaluation metrics
    try:
        with open("reports/metrics/train_metrics.json") as f:
            train_metrics = json.load(f)

        if "run_id" in train_metrics:
            with mlflow.start_run(run_id=train_metrics["run_id"]):
                mlflow.log_metrics(metrics)
                mlflow.log_artifact("reports/figures/predictions_vs_actual.csv")
                mlflow.log_artifact("reports/figures/residuals.csv")
                logger.info(f"Logged evaluation to run {train_metrics['run_id']}")
    except Exception as e:
        logger.warning(f"Could not log to existing run: {e}")

    logger.info("Evaluation complete!")
    return metrics


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for Hydra."""
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    evaluate_model(cfg)


def run_with_params():
    """Run with params.yaml for DVC compatibility."""
    params = load_params()
    cfg = OmegaConf.create(params)
    evaluate_model(cfg)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and (sys.argv[1].startswith("+") or sys.argv[1].startswith("-")):
        main()
    else:
        run_with_params()