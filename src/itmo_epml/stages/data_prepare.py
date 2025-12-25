"""Data preparation stage."""

import json
import logging
from pathlib import Path
import yaml

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params() -> dict:
    """Load parameters from params.yaml for DVC compatibility."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def prepare_data(cfg: DictConfig) -> dict:
    """
    Prepare raw data for feature engineering.

    Steps:
    1. Load raw data
    2. Basic cleaning (remove duplicates, handle obvious errors)
    3. Save interim data
    4. Generate data statistics
    """
    logger.info("Starting data preparation...")

    # Create directories
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # Load data
    train_path = cfg.data.train_path
    test_path = cfg.data.test_path

    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)

    test_df = None
    if Path(test_path).exists():
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

    # Basic statistics before cleaning - convert to native Python types
    stats = {
        "train_rows_original": int(len(train_df)),
        "train_cols": int(len(train_df.columns)),
        "train_duplicates": int(train_df.duplicated().sum()),
        "train_missing_total": int(train_df.isnull().sum().sum()),
    }

    # Remove duplicates
    train_df = train_df.drop_duplicates()
    stats["train_rows_after_dedup"] = int(len(train_df))

    # Log missing values per column
    missing_per_col = train_df.isnull().sum()
    stats["columns_with_missing"] = int((missing_per_col > 0).sum())
    stats["missing_percentage"] = {
        col: float(missing_per_col[col] / len(train_df) * 100)
        for col in missing_per_col[missing_per_col > 0].index[:10]  # Top 10
    }

    if test_df is not None:
        stats["test_rows"] = int(len(test_df))
        test_df = test_df.drop_duplicates()
        stats["test_rows_after_dedup"] = int(len(test_df))

    # Save interim data
    train_df.to_csv("data/interim/train_prepared.csv", index=False)
    logger.info("Saved prepared training data")

    if test_df is not None:
        test_df.to_csv("data/interim/test_prepared.csv", index=False)
        logger.info("Saved prepared test data")
    else:
        # Create empty file for DVC
        pd.DataFrame().to_csv("data/interim/test_prepared.csv", index=False)

    # Save statistics
    with open("reports/metrics/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Data preparation complete. Stats: {stats}")
    return stats


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for Hydra."""
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    prepare_data(cfg)


def run_with_params():
    """Run with params.yaml for DVC compatibility."""
    params = load_params()
    cfg = OmegaConf.create(params)
    prepare_data(cfg)


if __name__ == "__main__":
    # Check if running via DVC or Hydra
    import sys
    if len(sys.argv) > 1 and (sys.argv[1].startswith("+") or sys.argv[1].startswith("-")):
        main()
    else:
        run_with_params()