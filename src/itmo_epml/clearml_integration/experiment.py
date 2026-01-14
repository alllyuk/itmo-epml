"""ClearML experiment tracking utilities."""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from clearml import Dataset, Logger, Task

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Comprehensive experiment tracker using ClearML."""

    def __init__(self, task: Optional[Task] = None):
        """
        Initialize experiment tracker.

        Args:
            task: ClearML Task instance. If None, uses current task.
        """
        self.task = task or Task.current_task()
        if not self.task:
            raise RuntimeError("No active ClearML task. Call init_clearml first.")
        self.logger = self.task.get_logger()
        self._start_time = datetime.now()

    def log_parameters(self, params: Dict[str, Any], prefix: str = "") -> None:
        """
        Log parameters to ClearML.

        Args:
            params: Dictionary of parameters
            prefix: Optional prefix for parameter names
        """
        if prefix:
            params = {f"{prefix}/{k}": v for k, v in params.items()}

        self.task.connect(params)
        logger.info(f"Logged {len(params)} parameters")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        iteration: Optional[int] = None,
        series: str = "metrics",
    ) -> None:
        """
        Log metrics to ClearML.

        Args:
            metrics: Dictionary of metric name to value
            iteration: Optional iteration number
            series: Series name for grouping
        """
        for name, value in metrics.items():
            self.logger.report_scalar(
                title=series,
                series=name,
                value=value,
                iteration=iteration or 0,
            )
        logger.info(f"Logged metrics: {list(metrics.keys())}")

    def log_metric_series(
        self,
        name: str,
        values: List[float],
        title: str = "Training",
    ) -> None:
        """Log a series of metric values (e.g., loss over epochs)."""
        for i, value in enumerate(values):
            self.logger.report_scalar(
                title=title,
                series=name,
                value=value,
                iteration=i,
            )

    def log_confusion_matrix(
        self,
        matrix: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        iteration: int = 0,
    ) -> None:
        """Log confusion matrix visualization."""
        self.logger.report_confusion_matrix(
            title=title,
            series="confusion",
            matrix=matrix,
            xlabels=labels,
            ylabels=labels,
            iteration=iteration,
        )

    def log_scatter_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str = "Scatter Plot",
        series: str = "data",
        x_label: str = "X",
        y_label: str = "Y",
    ) -> None:
        """Log scatter plot."""
        self.logger.report_scatter2d(
            title=title,
            series=series,
            scatter=np.column_stack([x, y]),
            xaxis=x_label,
            yaxis=y_label,
        )

    def log_histogram(
        self,
        values: np.ndarray,
        title: str = "Histogram",
        series: str = "distribution",
        iteration: int = 0,
    ) -> None:
        """Log histogram."""
        self.logger.report_histogram(
            title=title,
            series=series,
            iteration=iteration,
            values=values,
            mode="relative",
        )

    def log_artifact(
        self,
        name: str,
        artifact_path: Union[str, Path],
        artifact_type: str = "artifact",
    ) -> None:
        """
        Log artifact file to ClearML.

        Args:
            name: Artifact name
            artifact_path: Path to artifact file
            artifact_type: Type of artifact (artifact, model, etc.)
        """
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            logger.warning(f"Artifact not found: {artifact_path}")
            return

        self.task.upload_artifact(name=name, artifact_object=str(artifact_path))
        logger.info(f"Uploaded artifact: {name}")

    def log_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        iteration: int = 0,
    ) -> None:
        """Log pandas DataFrame as table."""
        self.logger.report_table(
            title=name,
            series="data",
            table_plot=df,
            iteration=iteration,
        )

    def log_image(
        self,
        title: str,
        image_path: Union[str, Path],
        series: str = "images",
        iteration: int = 0,
    ) -> None:
        """Log image file."""
        self.logger.report_image(
            title=title,
            series=series,
            local_path=str(image_path),
            iteration=iteration,
        )

    def log_text(self, text: str, title: str = "Log") -> None:
        """Log text message."""
        self.logger.report_text(text, level=logging.INFO)

    def log_feature_importance(
        self,
        feature_names: List[str],
        importances: List[float],
        title: str = "Feature Importance",
    ) -> None:
        """Log feature importance as bar chart."""
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = [importances[i] for i in sorted_idx]

        # 1. Log as scatter plot (bar-like visualization)
        x_values = np.arange(len(sorted_values))
        self.logger.report_scatter2d(
            title=title,
            series="importance",
            scatter=np.column_stack([x_values, sorted_values]),
            xaxis="Feature Rank",
            yaxis="Importance",
        )

        # 2. Log as table for detailed inspection
        df = pd.DataFrame({
            'feature': sorted_names,
            'importance': sorted_values
        })
        self.log_dataframe(f"{title}_table", df, iteration=0)

    def set_tags(self, tags: List[str]) -> None:
        """Add tags to the task."""
        self.task.add_tags(tags)

    def get_task_id(self) -> str:
        """Get current task ID."""
        return self.task.id

    def get_task_url(self) -> str:
        """Get URL to task in ClearML web UI."""
        return self.task.get_output_log_web_page()


def create_experiment(
    project_name: str,
    task_name: str,
    task_type: str = "training",
    tags: Optional[List[str]] = None,
) -> ExperimentTracker:
    """
    Create a new ClearML experiment.

    Args:
        project_name: Name of the project
        task_name: Name of the task/experiment
        task_type: Type of task (training, testing, inference, etc.)
        tags: Optional list of tags

    Returns:
        ExperimentTracker instance
    """
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=getattr(Task.TaskTypes, task_type, Task.TaskTypes.training),
        tags=tags,
    )
    return ExperimentTracker(task)


@contextmanager
def experiment_context(
    project_name: str,
    task_name: str,
    task_type: str = "training",
    tags: Optional[List[str]] = None,
):
    """
    Context manager for ClearML experiments.

    Usage:
        with experiment_context("MyProject", "experiment_1") as tracker:
            tracker.log_metrics({"accuracy": 0.95})
    """
    tracker = create_experiment(project_name, task_name, task_type, tags)
    try:
        yield tracker
    finally:
        tracker.task.close()


def log_metrics(metrics: Dict[str, float], task: Optional[Task] = None) -> None:
    """Convenience function to log metrics to current task."""
    task = task or Task.current_task()
    if task:
        tracker = ExperimentTracker(task)
        tracker.log_metrics(metrics)


def log_artifacts(
    artifacts: Dict[str, Union[str, Path]],
    task: Optional[Task] = None,
) -> None:
    """Convenience function to log multiple artifacts."""
    task = task or Task.current_task()
    if task:
        tracker = ExperimentTracker(task)
        for name, path in artifacts.items():
            tracker.log_artifact(name, path)


class ExperimentComparator:
    """Compare multiple ClearML experiments."""

    def __init__(self, project_name: str):
        """
        Initialize comparator.

        Args:
            project_name: Name of the ClearML project
        """
        self.project_name = project_name

    def get_experiments(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Task]:
        """
        Get experiments from project.

        Args:
            tags: Filter by tags
            status: Filter by status (completed, running, etc.)
            limit: Maximum number of experiments to return

        Returns:
            List of Task objects
        """
        tasks = Task.get_tasks(
            project_name=self.project_name,
            tags=tags,
            task_filter={"status": [status]} if status else None,
        )
        return tasks[:limit]

    def compare_metrics(
        self,
        metric_names: List[str],
        tags: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare specific metrics across experiments.

        Args:
            metric_names: List of metric names to compare
            tags: Optional tags to filter experiments

        Returns:
            DataFrame with comparison results
        """
        tasks = self.get_experiments(tags=tags)

        comparison_data = []
        for task in tasks:
            row = {
                "task_id": task.id,
                "task_name": task.name,
                "status": task.status,
                "created": task.data.created,
            }

            # Get last reported metrics
            metrics = task.get_last_scalar_metrics()
            for metric_name in metric_names:
                for title, series_dict in metrics.items():
                    for series, value_dict in series_dict.items():
                        if series == metric_name or metric_name in series:
                            row[f"metric_{series}"] = value_dict.get("last", None)

            # Get parameters
            params = task.get_parameters()
            for param_name, param_value in params.items():
                row[f"param_{param_name}"] = param_value

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_best_experiment(
        self,
        metric_name: str,
        mode: str = "max",
        tags: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get best experiment based on a metric.

        Args:
            metric_name: Metric to optimize
            mode: 'max' or 'min'
            tags: Optional tags filter

        Returns:
            Dictionary with best experiment info
        """
        df = self.compare_metrics([metric_name], tags=tags)

        if df.empty:
            return None

        metric_col = f"metric_{metric_name}"
        if metric_col not in df.columns:
            # Try to find partial match
            matching_cols = [c for c in df.columns if metric_name in c]
            if matching_cols:
                metric_col = matching_cols[0]
            else:
                return None

        df_valid = df.dropna(subset=[metric_col])
        if df_valid.empty:
            return None

        if mode == "max":
            best_idx = df_valid[metric_col].idxmax()
        else:
            best_idx = df_valid[metric_col].idxmin()

        return df_valid.loc[best_idx].to_dict()

    def generate_comparison_report(
        self,
        metric_names: List[str],
        output_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Generate markdown comparison report."""
        df = self.compare_metrics(metric_names, tags=tags)

        report = f"# Experiment Comparison Report\n\n"
        report += f"**Project:** {self.project_name}\n"
        report += f"**Total Experiments:** {len(df)}\n\n"

        report += "## Experiments Overview\n\n"
        if not df.empty:
            report += df.to_markdown(index=False) + "\n\n"

        # Best experiments per metric
        report += "## Best Experiments\n\n"
        for metric in metric_names:
            best = self.get_best_experiment(metric, mode="max", tags=tags)
            if best:
                report += f"### Best by {metric}\n"
                report += f"- Task: {best.get('task_name', 'N/A')}\n"
                report += f"- ID: {best.get('task_id', 'N/A')}\n"
                report += f"- Value: {best.get(f'metric_{metric}', 'N/A')}\n\n"

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report