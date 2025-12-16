import itertools
import logging
from copy import deepcopy

import yaml
from itmo_epml.main import run_training_pipeline
from itmo_epml.model_registry import ModelRegistry
from mlops_utils import mlflow_experiment_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_hyperparameter_grid():
    """Генерирует сетку гиперпараметров для >15 экспериментов."""
    # 3 * 3 * 2 = 18 экспериментов
    grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5],
    }
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return experiments


def main():
    with open("configs/training_config.yaml") as f:
        base_config = yaml.safe_load(f)

    tracking_uri = base_config["mlflow"]["tracking_uri"]
    experiment_name = base_config["mlflow"]["experiment_name"]

    param_list = generate_hyperparameter_grid()
    logger.info(f"Prepared {len(param_list)} experiments to run.")

    for i, params in enumerate(param_list):
        run_name = f"run_{i+1}_est{params['n_estimators']}_depth{params['max_depth']}"
        logger.info(f"Starting experiment {i+1}/{len(param_list)}: {run_name}")

        current_config = deepcopy(base_config)
        current_config["model"]["hyperparameters"] = params
        current_config["mlflow"]["run_name"] = run_name

        try:
            with mlflow_experiment_context(experiment_name, run_name):
                # use_context=False, т.к. мы уже внутри контекста
                model, metrics, run_id = run_training_pipeline(
                    current_config, use_context=False
                )

                if metrics.get("val_r2", 0) > 0.5:  # фильтрация
                    registry = ModelRegistry(tracking_uri)
                    registry.register_model(
                        run_id=run_id,
                        model_name="HousePriceModel",
                        tags={"experiment_type": "grid_search", "params": str(params)},
                    )

        except Exception as e:
            logger.error(f"Experiment {run_name} failed: {e}")

    logger.info("Generating comparison report...")
    registry = ModelRegistry(tracking_uri)

    best_model = registry.get_best_model("HousePriceModel", metric="val_r2", mode="max")
    print("\n=== Best Model Found ===")
    print(best_model)

    report_path = r"reports\experiment_report.md"
    registry.generate_comparison_report("HousePriceModel", output_path=report_path)
    logger.info(f"Full report saved to {report_path}")


if __name__ == "__main__":
    main()
