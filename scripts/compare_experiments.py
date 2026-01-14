#!/usr/bin/env python3
"""Compare ClearML experiments and generate reports."""

import argparse
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_env():
    os.environ["CLEARML_API_HOST"] = "http://localhost:8008"
    os.environ["CLEARML_WEB_HOST"] = "http://localhost:8080"
    os.environ["CLEARML_FILES_HOST"] = "http://localhost:8081"


def compare_experiments(project_name: str = "House Prices Prediction"):
    """Compare all experiments in project."""
    from src.itmo_epml.clearml_integration.experiment import ExperimentComparator

    setup_env()

    comparator = ExperimentComparator(project_name)

    # Compare by key metrics
    metrics = ["val_r2", "val_mse", "val_mae"]
    df = comparator.compare_metrics(metrics)

    if not df.empty:
        print("\n=== Experiment Comparison ===")
        print(df.to_string())

        # Get best experiment
        best = comparator.get_best_experiment("val_r2", mode="max")
        if best:
            print("\n=== Best Experiment (by val_r2) ===")
            for k, v in best.items():
                print(f"  {k}: {v}")

    # Generate report
    report_path = "reports/metrics/experiment_comparison.md"
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    comparator.generate_comparison_report(metrics, output_path=report_path)
    logger.info(f"Report saved to: {report_path}")


def compare_models(model_name: str = "HousePriceModel"):
    """Compare all model versions."""
    from src.itmo_epml.clearml_integration.model_registry import ClearMLModelRegistry

    setup_env()

    registry = ClearMLModelRegistry()

    df = registry.compare_models(model_name=model_name)

    if not df.empty:
        print("\n=== Model Versions ===")
        print(df.to_string())

        # Get best model
        best = registry.get_best_model(model_name, metric_name="val_r2", mode="max")
        if best:
            print(f"\n=== Best Model ===")
            print(f"  ID: {best.id}")
            print(f"  Name: {best.name}")

    # Generate report
    report_path = "reports/metrics/model_comparison.md"
    registry.generate_model_report(model_name, output_path=report_path)
    logger.info(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare ClearML experiments")
    parser.add_argument(
        "--type",
        choices=["experiments", "models", "all"],
        default="all",
        help="What to compare",
    )
    args = parser.parse_args()

    if args.type in ["experiments", "all"]:
        compare_experiments()

    if args.type in ["models", "all"]:
        compare_models()


if __name__ == "__main__":
    main()