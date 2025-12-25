"""Model training stage."""

import json
import logging
from pathlib import Path
from typing import Any
import yaml

import hydra
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params() -> dict:
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def validate_config(cfg: DictConfig) -> None:
    """Validate model configuration parameters."""
    model_cfg = cfg.model

    if "_validate" in model_cfg:
        validation_rules = model_cfg["_validate"]
        for param, rules in validation_rules.items():
            if param in model_cfg:
                value = model_cfg[param]
                if "min" in rules and value < rules["min"]:
                    raise ValueError(
                        f"Parameter {param}={value} is below minimum {rules['min']}"
                    )
                if "max" in rules and value > rules["max"]:
                    raise ValueError(
                        f"Parameter {param}={value} exceeds maximum {rules['max']}"
                    )

    logger.info("Configuration validation passed")


def get_model(cfg: DictConfig) -> Any:
    """Create model instance based on configuration."""
    model_type = cfg.model.type

    # Filter out meta keys
    model_params = {
        k: v for k, v in cfg.model.items()
        if k not in ["type", "_validate"]
    }

    if model_type == "RandomForestRegressor":
        return RandomForestRegressor(**model_params)
    elif model_type == "LinearRegression":
        return LinearRegression(**model_params)
    elif model_type == "GradientBoostingRegressor":
        return GradientBoostingRegressor(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(cfg: DictConfig) -> dict:
    """
    Train model with MLflow tracking.

    Steps:
    1. Load processed data
    2. Initialize MLflow
    3. Create and train model
    4. Calculate metrics
    5. Log everything to MLflow
    6. Save model and metrics
    """
    logger.info("Starting model training...")

    # Validate configuration
    validate_config(cfg)

    # Create directories
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Load data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    logger.info(f"Training data shape: {X_train.shape}")

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = cfg.mlflow.run_name or f"{cfg.model.type}_dvc_run"

    with mlflow.start_run(run_name=run_name):
        # Log configuration
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

        # Log parameters
        model_params = {
            k: v for k, v in cfg.model.items()
            if k not in ["type", "_validate"]
        }
        mlflow.log_params({"model_type": cfg.model.type})
        mlflow.log_params(model_params)

        # Create and train model
        model = get_model(cfg)
        logger.info(f"Training {cfg.model.type}...")
        model.fit(X_train, y_train)

        # Calculate metrics
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        metrics = {
            "train_mse": float(mean_squared_error(y_train, y_pred_train)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
            "train_r2": float(r2_score(y_train, y_pred_train)),
            "val_mse": float(mean_squared_error(y_val, y_pred_val)),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
            "val_mae": float(mean_absolute_error(y_val, y_pred_val)),
            "val_r2": float(r2_score(y_val, y_pred_val)),
        }

        # Log metrics
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")

        # Log model
        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_train.head(5)
            )

        # Feature importance
        if hasattr(model, "feature_importances_") and cfg.mlflow.log_feature_importance:
            importance = dict(zip(X_train.columns, model.feature_importances_.tolist()))
            importance_sorted = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
            )
            mlflow.log_dict(importance_sorted, "feature_importance.json")

            with open("reports/figures/feature_importance.json", "w") as f:
                json.dump(importance_sorted, f, indent=2)

        # Register model if configured
        if cfg.mlflow.register_model and metrics["val_r2"] > 0.5:
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, cfg.mlflow.model_name)
            logger.info(f"Model registered as {cfg.mlflow.model_name}")

        run_id = mlflow.active_run().info.run_id

    # Save model locally
    joblib.dump(model, "models/model.joblib")

    # Save metrics
    metrics["run_id"] = run_id
    with open("reports/metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete!")
    return metrics


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for Hydra."""
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    train_model(cfg)


def run_with_params():
    """Run with params.yaml for DVC compatibility."""
    params = load_params()
    cfg = OmegaConf.create(params)
    train_model(cfg)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and (sys.argv[1].startswith("+") or sys.argv[1].startswith("-")):
        main()
    else:
        run_with_params()