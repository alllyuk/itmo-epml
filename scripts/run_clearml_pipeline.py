#!/usr/bin/env python3
"""Run ClearML ML Pipeline."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from clearml import Task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def setup_clearml_env():
    """Setup ClearML environment variables."""
    os.environ["CLEARML_API_HOST"] = "http://localhost:8008"
    os.environ["CLEARML_WEB_HOST"] = "http://localhost:8080"
    os.environ["CLEARML_FILES_HOST"] = "http://localhost:8081"


def run_pipeline_locally(config: dict):
    """Run the pipeline locally with ClearML tracking."""
    from src.itmo_epml.clearml_integration.experiment import ExperimentTracker
    from src.itmo_epml.clearml_integration.model_registry import ClearMLModelRegistry

    setup_clearml_env()

    # Initialize main task
    task = Task.init(
        project_name="House Prices Prediction",
        task_name=f"pipeline_run_{config['model']['type']}",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )

    tracker = ExperimentTracker(task)
    tracker.set_tags(["pipeline", "local_run", config["model"]["type"]])

    # Log configuration
    tracker.log_parameters(config, prefix="config")

    try:
        # Stage 1: Data Preparation
        logger.info("Stage 1: Data Preparation")
        from src.itmo_epml.stages.data_prepare import prepare_data
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(config)
        data_stats = prepare_data(cfg)
        tracker.log_metrics(
            {f"data_{k}": v for k, v in data_stats.items() if isinstance(v, (int, float))},
            series="data_stats"
        )

        # Stage 2: Feature Engineering
        logger.info("Stage 2: Feature Engineering")
        from src.itmo_epml.stages.feature_engineering import engineer_features

        feature_stats = engineer_features(cfg)
        tracker.log_metrics(
            {f"features_{k}": v for k, v in feature_stats.items()},
            series="feature_stats"
        )

        # Stage 3: Training
        logger.info("Stage 3: Model Training")
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        X_train = pd.read_csv("data/processed/X_train.csv")
        X_val = pd.read_csv("data/processed/X_val.csv")
        y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
        y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

        # Log training parameters
        model_params = {k: v for k, v in config["model"].items() if k not in ["type", "_validate"]}
        tracker.log_parameters(model_params, prefix="model")

        # Create and train model
        model_type = config["model"]["type"]
        if model_type == "RandomForestRegressor":
            model = RandomForestRegressor(**model_params)
        elif model_type == "GradientBoostingRegressor":
            model = GradientBoostingRegressor(**model_params)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)

        # Calculate metrics
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        train_metrics = {
            "train_mse": float(mean_squared_error(y_train, y_pred_train)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
            "train_r2": float(r2_score(y_train, y_pred_train)),
            "val_mse": float(mean_squared_error(y_val, y_pred_val)),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
            "val_mae": float(mean_absolute_error(y_val, y_pred_val)),
            "val_r2": float(r2_score(y_val, y_pred_val)),
        }

        tracker.log_metrics(train_metrics, series="training")
        logger.info(f"Training metrics: {train_metrics}")

        # Feature importance
        if hasattr(model, "feature_importances_"):
            tracker.log_feature_importance(
                feature_names=X_train.columns.tolist(),
                importances=model.feature_importances_.tolist(),
            )

        # Save model
        model_path = "models/model.joblib"
        joblib.dump(model, model_path)

        # Register model
        registry = ClearMLModelRegistry()
        output_model = registry.register_model(
            model_path=model_path,
            model_name="HousePriceModel",
            framework="scikit-learn",
            task=task,
            tags=["pipeline", config["model"]["type"]],
            metadata={
                "val_r2": train_metrics["val_r2"],
                "val_mse": train_metrics["val_mse"],
                "model_type": config["model"]["type"],
                **model_params,
            },
        )

        logger.info(f"Model registered: {output_model.id}")

        # Stage 4: Evaluation
        logger.info("Stage 4: Evaluation")
        residuals = y_val - y_pred_val
        eval_metrics = {
            "eval_mse": train_metrics["val_mse"],
            "eval_rmse": train_metrics["val_rmse"],
            "eval_mae": train_metrics["val_mae"],
            "eval_r2": train_metrics["val_r2"],
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
        }

        tracker.log_metrics(eval_metrics, series="evaluation")

        # Log plots
        tracker.log_scatter_plot(
            x=y_val,
            y=y_pred_val,
            title="Predictions vs Actual",
            x_label="Actual",
            y_label="Predicted",
        )

        tracker.log_histogram(
            values=residuals,
            title="Residuals Distribution",
        )

        # Save metrics
        Path("reports/metrics").mkdir(parents=True, exist_ok=True)
        with open("reports/metrics/clearml_metrics.json", "w") as f:
            json.dump({**train_metrics, **eval_metrics, "task_id": task.id}, f, indent=2)

        logger.info("Pipeline completed successfully!")
        logger.info(f"View results at: {task.get_output_log_web_page()}")

        return task.id, train_metrics

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        task.mark_failed(f"Pipeline error: {e}")
        raise

    finally:
        task.close()


def run_pipeline_with_decorator(config: dict):
    """Run pipeline using ClearML PipelineDecorator."""
    from src.itmo_epml.clearml_integration.pipeline import house_prices_pipeline

    setup_clearml_env()

    # Run decorated pipeline
    eval_metrics = house_prices_pipeline(config)
    return eval_metrics


def run_experiments_grid(config: dict, n_experiments: int = 15):
    """Run multiple experiments with different hyperparameters."""
    import itertools

    setup_clearml_env()

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "learning_rate": [0.05, 0.1],
    }

    keys, values = zip(*param_grid.items())
    combinations = list(itertools.product(*values))[:n_experiments]

    results = []
    for i, params in enumerate(combinations):
        param_dict = dict(zip(keys, params))
        run_name = f"experiment_{i+1}_est{param_dict['n_estimators']}_depth{param_dict['max_depth']}"

        logger.info(f"Running experiment {i+1}/{len(combinations)}: {run_name}")

        # Update config
        current_config = config.copy()
        current_config["model"] = {**current_config["model"], **param_dict}

        try:
            task_id, metrics = run_pipeline_locally(current_config)
            results.append({
                "experiment": run_name,
                "task_id": task_id,
                "params": param_dict,
                "metrics": metrics,
            })
        except Exception as e:
            logger.error(f"Experiment {run_name} failed: {e}")
            results.append({
                "experiment": run_name,
                "error": str(e),
            })

    # Save results
    with open("reports/metrics/experiments_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Completed {len(results)} experiments")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ClearML ML Pipeline")
    parser.add_argument(
        "--mode",
        choices=["local", "decorator", "grid"],
        default="local",
        help="Execution mode",
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=15,
        help="Number of experiments for grid mode",
    )
    args = parser.parse_args()

    config = load_config()

    if args.mode == "local":
        run_pipeline_locally(config)
    elif args.mode == "decorator":
        run_pipeline_with_decorator(config)
    elif args.mode == "grid":
        run_experiments_grid(config, args.experiments)


if __name__ == "__main__":
    main()