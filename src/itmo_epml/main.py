"""Model training with MLflow tracking."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops_utils import autolog_params, mlflow_experiment_context

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(train_path: str, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load training and test data."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train path {train_path} does not exist")

    train_df = pd.read_csv(train_path)
    test_df = None

    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)

    return train_df, test_df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "target",
    is_train: bool = True
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Basic feature preparation: Drop IDs, separate Target."""

    df_clean = df.copy()

    if "Id" in df_clean.columns:
        df_clean = df_clean.drop("Id", axis=1)

    y = None
    if target_col in df_clean.columns:
        y = df_clean[target_col].values
        df_clean = df_clean.drop(target_col, axis=1)
    elif is_train:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    return df_clean, y


def process_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """
    Process features ensuring.
    1. Encode Categoricals (Fit logic on Train, Apply to Val/Test)
    2. Impute Missing Values (Fit on Train)
    3. Scale Features (Fit on Train)
    """

    X_train_encoded = pd.get_dummies(X_train, drop_first=True, dtype=float)
    train_cols = X_train_encoded.columns.tolist()

    def align_columns(df_in: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Aligns dataframe columns to match training data structure."""
        if df_in is None:
            return None
        df_enc = pd.get_dummies(df_in, drop_first=False, dtype=float)

        # Reindex is the safest way to align columns:
        # - Keeps columns that exist in both
        # - Adds missing columns (filled with 0)
        # - Drops extra columns (that weren't in train)
        return df_enc.reindex(columns=train_cols, fill_value=0)

    X_val_encoded = align_columns(X_val)
    X_test_encoded = align_columns(X_test)

    imputer = SimpleImputer(strategy='mean')

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_encoded),
        columns=train_cols
    )
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val_encoded),
        columns=train_cols
    )

    X_test_imputed = None
    if X_test_encoded is not None:
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test_encoded),
            columns=train_cols
        )

    scaler = StandardScaler()

    X_train_scaled_arr = scaler.fit_transform(X_train_imputed)
    X_val_scaled_arr = scaler.transform(X_val_imputed)

    X_train_final = pd.DataFrame(X_train_scaled_arr, columns=train_cols)
    X_val_final = pd.DataFrame(X_val_scaled_arr, columns=train_cols)

    X_test_final = None
    if X_test_imputed is not None:
        X_test_final_arr = scaler.transform(X_test_imputed)
        X_test_final = pd.DataFrame(X_test_final_arr, columns=train_cols)

    return X_train_final, X_val_final, X_test_final, scaler

@autolog_params(exclude=["X_train", "y_train", "X_val", "y_val", "model"])
def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Train ML model and calculate metrics."""

    model.fit(X_train, y_train)

    metrics = {}

    # Training metrics
    y_pred_train = model.predict(X_train)
    metrics["train_mse"] = float(mean_squared_error(y_train, y_pred_train))
    metrics["train_mae"] = float(mean_absolute_error(y_train, y_pred_train))
    metrics["train_r2"] = float(r2_score(y_train, y_pred_train))

    # Validation metrics
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val)
        metrics["val_mse"] = float(mean_squared_error(y_val, y_pred_val))
        metrics["val_mae"] = float(mean_absolute_error(y_val, y_pred_val))
        metrics["val_r2"] = float(r2_score(y_val, y_pred_val))

    return model, metrics


def run_training_pipeline(
        config: Dict[str, Any],
        use_context: bool = True
    ) -> Tuple[Any, Dict, str]:
    """Execute training pipeline based on configuration dict."""

    # Setup MLflow Config
    mlflow_config = config.get("mlflow", {})
    mlflow_uri = mlflow_config.get("tracking_uri", "file:///mlruns")
    exp_name = mlflow_config.get("experiment_name", "default_experiment")
    run_name = mlflow_config.get("run_name", "default_run")

    if mlflow_uri.startswith("file:///"):
        path_part = mlflow_uri.replace("file:///", "")
        Path(path_part).mkdir(parents=True, exist_ok=True)
        mlflow_uri = f"file:///{Path(path_part).absolute()}"

    mlflow.set_tracking_uri(mlflow_uri)

    def _execute_logic():
        mlflow.log_dict(config, "config.yaml")

        logger.info(f"Loading data from {config['data']['train_path']}")
        train_df_raw, test_df_raw = load_data(
            config['data'].get('train_path'),
            config['data'].get('test_path')
        )

        target_col = config['data'].get("target_col", "target")
        test_size = config['data'].get("test_size", 0.2)
        random_state = config['data'].get("random_state", 42)

        X_full, y_full = prepare_features(train_df_raw, target_col=target_col, is_train=True)

        X_test_raw = None
        if test_df_raw is not None:
            X_test_raw, _ = prepare_features(test_df_raw, target_col=target_col, is_train=False)

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_full, y_full, test_size=test_size, random_state=random_state
        )

        X_train, X_val, X_test, scaler = process_features(X_train_raw, X_val_raw, X_test_raw)

        model_config = config.get("model", {})
        if "hyperparameters" in model_config:
            model_params = model_config["hyperparameters"]
        else:
            model_params = {k: v for k, v in model_config.items() if k != "type"}


        logger.info(f"Training model with params: {model_params}")
        mlflow.log_params(model_params)

        if model_config.get('type') == 'RandomForestRegressor':
            model = RandomForestRegressor(**model_params)
        elif model_config.get('type') == 'LinearRegression':
            model = LinearRegression(**model_params)
        else:
            logger.warning(f"Model type {model_config.get('type')} not found. Defaulting to RandomForest.")
            model = RandomForestRegressor(**model_params)

        model, metrics = train_model(
            model, X_train, y_train, X_val, y_val
        )

        logger.info(f"Metrics: {metrics}")
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="models",
            input_example=X_train.head(5)
        )

        if hasattr(model, "feature_importances_"):
            feature_importance = {
                col: float(imp)
                for col, imp in zip(X_train.columns, model.feature_importances_)
            }
            mlflow.log_dict(feature_importance, "feature_importance.json")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run completed. ID: {run_id}")

        return model, metrics, run_id

    if use_context:
        with mlflow_experiment_context(exp_name, run_name):
            return _execute_logic()
    else:
        # Если контекст уже создан снаружи (в цикле экспериментов)
        return _execute_logic()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        run_training_pipeline(config)
    else:
        logger.error(f"Config file not found: {args.config}")
