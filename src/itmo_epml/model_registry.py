"""Model registry and comparison utilities."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Model registry and versioning management."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize Model Registry."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "models",
        description: Optional[str] = None,
        tags: Optional[dict] = None,
    ):
        """Register a model from MLflow run."""
        try:
            # Log model details
            model_uri = f"runs:/{run_id}/{artifact_path}"

            # Register model
            result = mlflow.register_model(model_uri, model_name)

            logger.info(f"Model {model_name} registered successfully")

            # Update description and tags
            if description:
                self.client.update_registered_model(model_name, description)

            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        model_name, result.version, key, value
                    )

            return result
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def get_model_versions(self, model_name: str) -> list:
        """Get all versions of a registered model."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return versions
        except Exception as e:
            logger.error(f"Error retrieving model versions: {e}")
            return []

    def compare_models(self, model_name: str) -> pd.DataFrame:
        """Compare different model versions and their metrics."""
        try:
            versions = self.get_model_versions(model_name)
            comparison_data = []

            for version in versions:
                run_id = version.run_id
                try:
                    run = self.client.get_run(run_id)
                except Exception:
                    logger.warning(f"Run {run_id} not found for version {version.version}")
                    continue

                # Get metrics and params
                metrics = run.data.metrics
                params = run.data.params

                row = {
                    "Version": version.version,
                    "Status": version.status,
                    "Run ID": run_id,
                    "Created": pd.Timestamp(version.creation_timestamp, unit="ms"),
                }

                # Add all metrics with prefix
                for metric_name, metric_value in metrics.items():
                    row[f"metric_{metric_name}"] = metric_value

                # Add all params with prefix
                for param_name, param_value in params.items():
                    row[f"param_{param_name}"] = param_value

                comparison_data.append(row)

            df = pd.DataFrame(comparison_data)

            # Sort by creation time
            if not df.empty and "Created" in df.columns:
                df = df.sort_values("Created", ascending=False)

            return df
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()

    def get_best_model(
            self,
            model_name: str,
            metric: str = "val_r2",
            mode: str = "max"
        ) -> Optional[Dict[str, Any]]:
        """
        Get best model version based on a metric.

        Args:
            model_name: Name of the registered model
            metric: Metric name to compare (without 'metric_' prefix)
            mode: 'max' for higher is better (accuracy), 'min' for lower is better (MSE, Loss)
        """
        try:
            comparison_df = self.compare_models(model_name)

            if comparison_df.empty:
                logger.warning(f"No model versions found for {model_name}")
                return None

            metric_col = f"metric_{metric}"
            if metric_col not in comparison_df.columns:
                logger.warning(f"Metric {metric} not found in comparison data")
                return None

            # Drop NaNs for the metric calculation
            valid_df = comparison_df.dropna(subset=[metric_col])

            if valid_df.empty:
                logger.warning(f"All values for metric {metric} are NaN")
                return None

            if mode == "max":
                best_idx = valid_df[metric_col].idxmax()
            elif mode == "min":
                best_idx = valid_df[metric_col].idxmin()
            else:
                raise ValueError("Mode must be 'min' or 'max'")

            best_row = valid_df.loc[best_idx]
            return best_row.to_dict()

        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return None

    def list_experiments(self, name_pattern: Optional[str] = None) -> pd.DataFrame:
        """
        List experiments with optional name filtering.

        Args:
            name_pattern: Optional string to filter experiment names (SQL LIKE syntax, e.g., '%CNN%')
        """
        try:
            filter_string = None
            if name_pattern:
                # MLflow supports ILIKE for case-insensitive search
                filter_string = f"name ILIKE '%{name_pattern}%'"

            experiments = self.client.search_experiments(
                filter_string=filter_string,
                view_type=ViewType.ACTIVE_ONLY,
                order_by=["creation_time DESC"]
            )

            exp_data = [
                {
                    "Experiment ID": exp.experiment_id,
                    "Name": exp.name,
                    "Artifact Location": exp.artifact_location,
                    "Created": pd.Timestamp(exp.creation_time, unit="ms") if exp.creation_time else None,
                }
                for exp in experiments
            ]
            return pd.DataFrame(exp_data)
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return pd.DataFrame()

    def list_runs(
        self,
        experiment_id: str,
        filter_string: str = "",
        order_by: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        List runs in an experiment with filtering and sorting.

        Args:
            experiment_id: ID of the experiment
            filter_string: MLflow search filter (e.g., "metrics.rmse < 1.0 and params.model = 'tree'")
            order_by: List of columns to order by (e.g., ["metrics.rmse ASC"])
        """
        try:
            # If no order specified, default to start_time desc
            if order_by is None:
                order_by = ["start_time DESC"]

            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                order_by=order_by
            )

            runs_data = []
            for run in runs:
                data = {
                    "Run ID": run.info.run_id,
                    "Run Name": run.info.run_name,
                    "Status": run.info.status,
                    "Created": pd.Timestamp(run.info.start_time, unit="ms"),
                    "User": run.info.user_id,
                }

                # Flatten metrics
                for k, v in run.data.metrics.items():
                    data[f"metric_{k}"] = v

                # Flatten params
                for k, v in run.data.params.items():
                    data[f"param_{k}"] = v

                # Flatten tags (optional, selecting common ones)
                if "mlflow.source.name" in run.data.tags:
                    data["source"] = run.data.tags["mlflow.source.name"]

                runs_data.append(data)

            return pd.DataFrame(runs_data)
        except Exception as e:
            logger.error(f"Error listing runs: {e}")
            return pd.DataFrame()

    def compare_experiment_runs(
        self,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        metric: str = "val_loss",
        ascending: bool = True,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Compare different models (runs) within one experiment.

        Args:
            experiment_id: ID of the experiment
            experiment_name: Name of the experiment (used if ID is not provided)
            metric: Metric name to sort by (without 'metric_' prefix)
            ascending: True for metrics where lower is better (Loss), False for higher (Accuracy/R2)
            top_n: Number of top runs to return
        """
        try:
            # Resolve experiment ID
            if not experiment_id and experiment_name:
                exp = self.client.get_experiment_by_name(experiment_name)
                if exp:
                    experiment_id = exp.experiment_id

            if not experiment_id:
                raise ValueError("Must provide either experiment_id or experiment_name")

            # Get all runs
            df = self.list_runs(experiment_id)

            if df.empty:
                logger.warning(f"No runs found for experiment {experiment_id}")
                return pd.DataFrame()

            target_col = f"metric_{metric}"

            # Check if metric exists
            if target_col not in df.columns:
                logger.warning(f"Metric '{metric}' not found in experiment runs.")
                # Return unsorted or sorted by creation date
                return df.head(top_n)

            # Sort
            df_sorted = df.sort_values(by=target_col, ascending=ascending)

            return df_sorted.head(top_n)

        except Exception as e:
            logger.error(f"Error comparing experiment runs: {e}")
            return pd.DataFrame()

    def generate_comparison_report(self, model_name: str, output_path: Optional[str] = None) -> str:
        """Generate a comparison report for all model versions."""

        # Get comparison dataframe
        comparison_df = self.compare_models(model_name)

        if comparison_df.empty:
            return f"No model versions found for {model_name}"

        # Generate report
        report = f"# Model Comparison Report: {model_name}\n\n"
        report += comparison_df.to_markdown(index=False) + "\n\n"

        # Find best model
        best_model = self.get_best_model(model_name, metric="val_r2")
        if best_model:
            report += "## Best Model (Registered)\n"
            for key, value in best_model.items():
                report += f"- **{key}**: {value}\n"

        # Save report if path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    mlflow_dir = Path("mlruns")
    mlflow.set_tracking_uri(f"file:///{mlflow_dir.absolute()}")

    registry = ModelRegistry()

    # 1. Поиск экспериментов по паттерну
    print("\n--- Searching Experiments containing 'Test' ---")
    experiments = registry.list_experiments(name_pattern="Test")
    print(experiments)

    if not experiments.empty:
        exp_id = experiments.iloc[0]["Experiment ID"]

        # 2. Фильтрация запусков (runs)
        print(f"\n--- Searching Runs in Exp {exp_id} with low loss ---")
        # Пример синтаксиса MLflow: поиск параметров и метрик
        runs_df = registry.list_runs(
            experiment_id=exp_id,
            filter_string="metrics.loss < 0.5",
            order_by=["metrics.loss ASC"]
        )
        print(runs_df[["Run ID", "Status", "metric_loss"]].head() if "metric_loss" in runs_df.columns else runs_df.head())

        # 3. Сравнение всех моделей в эксперименте
        print(f"\n--- Comparing Top Models in Exp {exp_id} ---")
        comparison = registry.compare_experiment_runs(
            experiment_id=exp_id,
            metric="accuracy",
            ascending=False, # Higher accuracy is better
            top_n=3
        )
        print(comparison)