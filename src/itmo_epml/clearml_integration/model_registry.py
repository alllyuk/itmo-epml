"""ClearML Model Registry for model versioning and management."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from clearml import InputModel, Model, OutputModel, Task

logger = logging.getLogger(__name__)


class ClearMLModelRegistry:
    """Model registry using ClearML for versioning and management."""

    def __init__(self, project_name: str = "House Prices Prediction"):
        """
        Initialize model registry.

        Args:
            project_name: Name of the ClearML project
        """
        self.project_name = project_name

    def register_model(
        self,
        model_path: Union[str, Path],
        model_name: str,
        framework: str = "scikit-learn",
        task: Optional[Task] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_version: bool = True,
    ) -> OutputModel:
        """
        Register a model to ClearML Model Registry.

        Args:
            model_path: Path to the model file
            model_name: Name for the model
            framework: ML framework (scikit-learn, pytorch, tensorflow, etc.)
            task: ClearML task (uses current if None)
            tags: Optional tags for the model
            metadata: Optional metadata dictionary
            auto_version: Automatically version the model

        Returns:
            ClearML OutputModel instance
        """
        task = task or Task.current_task()
        if not task:
            raise RuntimeError("No active ClearML task")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create output model
        output_model = OutputModel(
            task=task,
            framework=framework,
            name=model_name,
        )

        # Add tags
        if tags:
            output_model.set_tags(tags)

        # Add metadata as labels
        if metadata:
            labels = {str(k): str(v) for k, v in metadata.items()}
            output_model.update_labels(labels)

        # Upload model
        output_model.update_weights(
            weights_filename=str(model_path),
            auto_delete_file=False,
            register_uri=None,
        )

        logger.info(f"Model registered: {model_name} (ID: {output_model.id})")
        return output_model

    def get_model(
        self,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        only_published: bool = False,
    ) -> Optional[Model]:
        """
        Get a model from registry.

        Args:
            model_id: Specific model ID
            model_name: Model name (gets latest version)
            tags: Filter by tags
            only_published: Only return published models

        Returns:
            ClearML Model instance or None
        """
        if model_id:
            return Model(model_id=model_id)

        # Search for model
        models = Model.query_models(
            project_name=self.project_name,
            model_name=model_name,
            tags=tags,
            only_published=only_published,
        )

        if not models:
            logger.warning(f"No models found matching criteria")
            return None

        # Return most recent
        return models[0]

    def get_model_versions(
        self,
        model_name: str,
        limit: int = 10,
    ) -> List[Model]:
        """
        Get all versions of a model.

        Args:
            model_name: Name of the model
            limit: Maximum number of versions to return

        Returns:
            List of Model instances
        """
        models = Model.query_models(
            project_name=self.project_name,
            model_name=model_name,
        )
        return models[:limit]

    def compare_models(
        self,
        model_name: Optional[str] = None,
        model_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple model versions.

        Args:
            model_name: Compare all versions of this model
            model_ids: Or compare specific model IDs

        Returns:
            DataFrame with comparison
        """
        if model_name:
            models = self.get_model_versions(model_name)
        elif model_ids:
            models = [Model(model_id=mid) for mid in model_ids]
        else:
            raise ValueError("Provide model_name or model_ids")

        comparison_data = []
        for model in models:
            row = {
                "model_id": model.id,
                "name": model.name,
                "framework": model.framework,
                "created": model.created,
                "published": model.published,
                "url": model.url,
            }

            # Get labels (metadata)
            labels = model.labels
            for key, value in labels.items():
                row[f"meta_{key}"] = value

            # Get associated task metrics
            if model.task:
                try:
                    task = Task.get_task(task_id=model.task)
                    metrics = task.get_last_scalar_metrics()
                    for title, series_dict in metrics.items():
                        for series, value_dict in series_dict.items():
                            row[f"metric_{series}"] = value_dict.get("last")
                except Exception as e:
                    logger.warning(f"Could not get task metrics: {e}")

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_best_model(
        self,
        model_name: str,
        metric_name: str = "val_r2",
        mode: str = "max",
    ) -> Optional[Model]:
        """
        Get best model version based on metric.

        Args:
            model_name: Name of the model
            metric_name: Metric to optimize
            mode: 'max' or 'min'

        Returns:
            Best Model instance
        """
        df = self.compare_models(model_name=model_name)

        if df.empty:
            return None

        metric_col = f"metric_{metric_name}"
        if metric_col not in df.columns:
            matching = [c for c in df.columns if metric_name in c]
            if matching:
                metric_col = matching[0]
            else:
                logger.warning(f"Metric {metric_name} not found")
                return None

        df_valid = df.dropna(subset=[metric_col])
        if df_valid.empty:
            return None

        if mode == "max":
            best_idx = df_valid[metric_col].idxmax()
        else:
            best_idx = df_valid[metric_col].idxmin()

        best_id = df_valid.loc[best_idx, "model_id"]
        return Model(model_id=best_id)

    def publish_model(self, model_id: str) -> bool:
        """
        Publish a model (mark as production-ready).

        Args:
            model_id: ID of the model to publish

        Returns:
            True if successful
        """
        try:
            model = Model(model_id=model_id)
            model.publish()
            logger.info(f"Model {model_id} published")
            return True
        except Exception as e:
            logger.error(f"Failed to publish model: {e}")
            return False

    def archive_model(self, model_id: str) -> bool:
        """
        Archive a model.

        Args:
            model_id: ID of the model to archive

        Returns:
            True if successful
        """
        try:
            model = Model(model_id=model_id)
            model.archive()
            logger.info(f"Model {model_id} archived")
            return True
        except Exception as e:
            logger.error(f"Failed to archive model: {e}")
            return False

    def download_model(
        self,
        model_id: str,
        target_path: Union[str, Path],
    ) -> str:
        """
        Download model weights to local path.

        Args:
            model_id: ID of the model
            target_path: Where to save the model

        Returns:
            Path to downloaded model
        """
        model = Model(model_id=model_id)
        local_path = model.get_local_copy(target_path)
        logger.info(f"Model downloaded to: {local_path}")
        return local_path

    def generate_model_report(
        self,
        model_name: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate markdown report for model versions."""
        df = self.compare_models(model_name=model_name)

        report = f"# Model Report: {model_name}\n\n"
        report += f"**Project:** {self.project_name}\n"
        report += f"**Total Versions:** {len(df)}\n\n"

        report += "## Model Versions\n\n"
        if not df.empty:
            report += df.to_markdown(index=False) + "\n\n"

        # Best model
        best = self.get_best_model(model_name, metric_name="val_r2", mode="max")
        if best:
            report += "## Best Model\n\n"
            report += f"- **ID:** {best.id}\n"
            report += f"- **Framework:** {best.framework}\n"
            report += f"- **Published:** {best.published}\n"

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report


# Convenience functions
def register_model(
    model_path: Union[str, Path],
    model_name: str,
    **kwargs,
) -> OutputModel:
    """Register model to ClearML registry."""
    registry = ClearMLModelRegistry()
    return registry.register_model(model_path, model_name, **kwargs)


def compare_models(model_name: str) -> pd.DataFrame:
    """Compare model versions."""
    registry = ClearMLModelRegistry()
    return registry.compare_models(model_name=model_name)