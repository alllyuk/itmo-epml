"""Model training with MLflow tracking."""

import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(train_path: str, test_path: Optional[str] = None):
    """Load training and test data."""
    train_df = pd.read_csv(train_path)

    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    return train_df, None


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame = None):
    """Preprocess data."""
    # Handle different target column names
    target_col = None
    for col in ["target", "SalePrice", "Price", "label"]:
        if col in train_df.columns:
            target_col = col
            break

    if target_col:
        X_train = train_df.drop([target_col, "Id"], axis=1, errors="ignore")
        y_train = train_df[target_col].values
    else:
        X_train = train_df.copy()
        y_train = None

    # Handle categorical columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)

    # Fill missing values
    X_train = X_train.fillna(X_train.mean())
    X_train = X_train.astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)

    if test_df is not None:
        if target_col and target_col in test_df.columns:
            X_test = test_df.drop([target_col, "Id"], axis=1, errors="ignore")
            y_test = test_df[target_col].values
        else:
            X_test = test_df.drop("Id", axis=1, errors="ignore")
            y_test = None

        # Apply same transformations
        test_categorical_cols = X_test.select_dtypes(include=["object"]).columns
        if len(test_categorical_cols) > 0:
            X_test = pd.get_dummies(
                X_test, columns=test_categorical_cols, drop_first=True
            )

        # Align columns with training data
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[X_train.columns]

        X_test = X_test.fillna(X_test.mean())
        X_test = X_test.astype(np.float32)

        X_test_scaled = scaler.transform(X_test.values)
        return X_train_scaled, y_train, X_test_scaled, y_test, scaler

    return X_train_scaled, y_train, None, None, scaler


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    n_estimators=100,
    max_depth=10,
    random_state=42,
):
    """Train Random Forest model."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

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


def main(
    train_path: str = "data/raw/train.csv",
    experiment_name: str = "default_experiment",
    run_name: str = "baseline_model",
    n_estimators: int = 100,
    max_depth: int = 10,
):
    """Main training pipeline with MLflow tracking."""

    # Set MLflow tracking URI
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlflow_dir.absolute()}")

    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        logger.info(f"Starting training run: {run_name}")

        # Load data
        logger.info(f"Loading data from {train_path}")
        train_df, test_df = load_data(train_path, "data/raw/test.csv")

        # Preprocess data
        logger.info("Preprocessing data")
        if test_df is not None:
            X_train_full, y_train_full, X_test, y_test, scaler = preprocess_data(
                train_df, test_df
            )
            # Split training data into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=42
            )
        else:
            X_train_full, y_train_full, _, _, scaler = preprocess_data(train_df)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=42
            )

        # Log parameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
            "test_size": 0.2,
        }
        mlflow.log_params(params)
        logger.info(f"Logged parameters: {params}")

        # Train model
        logger.info("Training Random Forest model")
        model, metrics = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

        # Log metrics
        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")

        # Log model
        logger.info("Logging model")
        mlflow.sklearn.log_model(model, "model", input_example=X_train[:5])

        # Log additional metadata
        metadata = {
            "data_source": train_path,
            "preprocessing": "StandardScaler",
            "model_type": "RandomForestClassifier",
        }
        mlflow.log_dict(metadata, "metadata.json")

        # Log feature importance
        feature_importance = {
            f"feature_{i}": float(importance)
            for i, importance in enumerate(model.feature_importances_)
        }
        mlflow.log_dict(feature_importance, "feature_importance.json")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Training completed. Run ID: {run_id}")

        return model, metrics, run_id


if __name__ == "__main__":
    main()
