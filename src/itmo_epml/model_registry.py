"""Model registry and comparison utilities."""

import logging
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

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
        description: Optional[str] = None,
        tags: Optional[dict] = None,
    ):
        """Register a model from MLflow run."""
        try:
            # Log model details
            model_uri = f"runs:/{run_id}/model"

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
                run = self.client.get_run(run_id)

                # Get metrics
                metrics = run.data.metrics
                params = run.data.params

                row = {
                    "Version": version.version,
                    "Status": version.status,
                    "Run ID": run_id,
                    "Created": version.creation_timestamp,
                }

                # Add metrics
                for metric_name, metric_value in metrics.items():
                    row[f"metric_{metric_name}"] = metric_value

                # Add key parameters
                for param_name in ["n_estimators", "max_depth"]:
                    if param_name in params:
                        row[f"param_{param_name}"] = params[param_name]

                comparison_data.append(row)

            df = pd.DataFrame(comparison_data)

            # Sort by creation time
            if "Created" in df.columns:
                df = df.sort_values("Created", ascending=False)

            return df
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()

    def get_best_model(self, model_name: str, metric: str = "val_accuracy") -> dict:
        """Get best model version based on a metric."""
        try:
            comparison_df = self.compare_models(model_name)

            metric_col = f"metric_{metric}"
            if metric_col not in comparison_df.columns:
                logger.warning(f"Metric {metric} not found in comparison data")
                return None

            best_row = comparison_df.loc[comparison_df[metric_col].idxmax()]
            return best_row.to_dict()
        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return None

    def list_experiments(self) -> pd.DataFrame:
        """List all experiments."""
        try:
            experiments = self.client.search_experiments()
            exp_data = [
                {
                    "Experiment ID": exp.experiment_id,
                    "Name": exp.name,
                    "Created": pd.Timestamp(exp.creation_time, unit="ms"),
                }
                for exp in experiments
            ]
            return pd.DataFrame(exp_data)
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return pd.DataFrame()

    def list_runs(self, experiment_id: str) -> pd.DataFrame:
        """List all runs in an experiment."""
        try:
            runs = self.client.search_runs(experiment_ids=[experiment_id])
            runs_data = [
                {
                    "Run ID": run.info.run_id,
                    "Run Name": run.info.run_name,
                    "Status": run.info.status,
                    "Created": pd.Timestamp(run.info.start_time, unit="ms"),
                    "Metrics": len(run.data.metrics),
                }
                for run in runs
            ]
            return pd.DataFrame(runs_data)
        except Exception as e:
            logger.error(f"Error listing runs: {e}")
            return pd.DataFrame()


def generate_comparison_report(model_name: str, output_path: Optional[str] = None) -> str:
    """Generate a comparison report for all model versions."""
    mlflow_dir = Path("mlruns")
    mlflow.set_tracking_uri(f"file:///{mlflow_dir.absolute()}")

    registry = ModelRegistry()

    # Get comparison dataframe
    comparison_df = registry.compare_models(model_name)

    if comparison_df.empty:
        return f"No model versions found for {model_name}"

    # Generate report
    report = f"# Model Comparison Report: {model_name}\n\n"
    report += comparison_df.to_markdown(index=False) + "\n\n"

    # Find best model
    best_model = registry.get_best_model(model_name, metric="val_accuracy")
    if best_model:
        report += "## Best Model\n"
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
    registry = ModelRegistry()
    experiments = registry.list_experiments()
    print("Available Experiments:")
    print(experiments)
