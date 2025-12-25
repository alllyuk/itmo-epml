"""Feature engineering stage."""

import json
import logging
from pathlib import Path
import yaml

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params() -> dict:
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def engineer_features(cfg: DictConfig) -> dict:
    """
    Feature engineering pipeline.

    Steps:
    1. Load prepared data
    2. Separate features and target
    3. Train/validation split
    4. Encode categoricals
    5. Impute missing values
    6. Scale features
    7. Save processed data and transformers
    """
    logger.info("Starting feature engineering...")

    # Create directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv("data/interim/train_prepared.csv")
    test_df = pd.read_csv("data/interim/test_prepared.csv")

    target_col = cfg.data.target_col
    test_size = cfg.data.test_size
    random_state = cfg.data.random_state

    # Separate features and target
    if "Id" in train_df.columns:
        train_df = train_df.drop("Id", axis=1)

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    y = train_df[target_col].values
    X = train_df.drop(target_col, axis=1)

    # Handle test data
    X_test_raw = None
    if not test_df.empty:
        if "Id" in test_df.columns:
            test_df = test_df.drop("Id", axis=1)
        if target_col in test_df.columns:
            test_df = test_df.drop(target_col, axis=1)
        X_test_raw = test_df

    # Train/validation split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train size: {len(X_train_raw)}, Val size: {len(X_val_raw)}")

    # Encode categoricals
    X_train_encoded = pd.get_dummies(X_train_raw, drop_first=True, dtype=float)
    feature_columns = X_train_encoded.columns.tolist()

    def align_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return None
        df_enc = pd.get_dummies(df, drop_first=False, dtype=float)
        return df_enc.reindex(columns=feature_columns, fill_value=0)

    X_val_encoded = align_columns(X_val_raw)
    X_test_encoded = align_columns(X_test_raw) if X_test_raw is not None else None

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_encoded),
        columns=feature_columns
    )
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val_encoded),
        columns=feature_columns
    )

    X_test_imputed = None
    if X_test_encoded is not None:
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test_encoded),
            columns=feature_columns
        )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=feature_columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_imputed),
        columns=feature_columns
    )

    X_test_scaled = None
    if X_test_imputed is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imputed),
            columns=feature_columns
        )

    # Save processed data
    X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
    X_val_scaled.to_csv("data/processed/X_val.csv", index=False)
    pd.DataFrame(y_train, columns=[target_col]).to_csv(
        "data/processed/y_train.csv", index=False
    )
    pd.DataFrame(y_val, columns=[target_col]).to_csv(
        "data/processed/y_val.csv", index=False
    )

    if X_test_scaled is not None:
        X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
    else:
        pd.DataFrame().to_csv("data/processed/X_test.csv", index=False)

    # Save transformers
    joblib.dump(scaler, "models/scaler.joblib")

    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    stats = {
        "n_features": len(feature_columns),
        "train_samples": len(X_train_scaled),
        "val_samples": len(X_val_scaled),
        "test_samples": len(X_test_scaled) if X_test_scaled is not None else 0,
    }

    logger.info(f"Feature engineering complete. Stats: {stats}")
    return stats


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for Hydra."""
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    engineer_features(cfg)


def run_with_params():
    """Run with params.yaml for DVC compatibility."""
    params = load_params()
    cfg = OmegaConf.create(params)
    engineer_features(cfg)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and (sys.argv[1].startswith("+") or sys.argv[1].startswith("-")):
        main()
    else:
        run_with_params()