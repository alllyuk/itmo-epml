"""ClearML Pipelines for ML workflow automation."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from clearml import PipelineController, PipelineDecorator, Task
from clearml.automation import TriggerScheduler

logger = logging.getLogger(__name__)


class ClearMLPipeline:
    """ClearML Pipeline for ML workflow automation."""

    def __init__(
        self,
        name: str,
        project: str = "House Prices Prediction",
        version: str = "1.0.0",
        default_queue: str = "default",
    ):
        """
        Initialize ClearML Pipeline.

        Args:
            name: Pipeline name
            project: Project name
            version: Pipeline version
            default_queue: Default execution queue
        """
        self.name = name
        self.project = project
        self.version = version
        self.default_queue = default_queue
        self.pipeline: Optional[PipelineController] = None

    def create_pipeline(
        self,
        add_pipeline_tags: bool = True,
        abort_on_failure: bool = True,
    ) -> PipelineController:
        """
        Create a new pipeline controller.

        Args:
            add_pipeline_tags: Add pipeline tags to steps
            abort_on_failure: Stop pipeline on step failure

        Returns:
            PipelineController instance
        """
        self.pipeline = PipelineController(
            name=self.name,
            project=self.project,
            version=self.version,
            add_pipeline_tags=add_pipeline_tags,
            abort_on_failure=abort_on_failure,
        )

        logger.info(f"Pipeline created: {self.name} v{self.version}")
        return self.pipeline

    def add_step(
        self,
        name: str,
        base_task_project: str,
        base_task_name: str,
        parameter_override: Optional[Dict[str, Any]] = None,
        parents: Optional[List[str]] = None,
        execution_queue: Optional[str] = None,
        cache_executed_step: bool = True,
    ) -> "ClearMLPipeline":
        """
        Add a step to the pipeline.

        Args:
            name: Step name
            base_task_project: Project containing base task
            base_task_name: Name of base task
            parameter_override: Parameters to override
            parents: Parent step names
            execution_queue: Queue for execution
            cache_executed_step: Cache results

        Returns:
            Self for chaining
        """
        if not self.pipeline:
            self.create_pipeline()

        self.pipeline.add_step(
            name=name,
            base_task_project=base_task_project,
            base_task_name=base_task_name,
            parameter_override=parameter_override or {},
            parents=parents or [],
            execution_queue=execution_queue or self.default_queue,
            cache_executed_step=cache_executed_step,
        )

        logger.info(f"Added step: {name}")
        return self

    def add_function_step(
        self,
        name: str,
        function: Callable,
        function_kwargs: Optional[Dict[str, Any]] = None,
        parents: Optional[List[str]] = None,
        execution_queue: Optional[str] = None,
        packages: Optional[List[str]] = None,
    ) -> "ClearMLPipeline":
        """
        Add a function step to the pipeline.

        Args:
            name: Step name
            function: Python function to execute
            function_kwargs: Function arguments
            parents: Parent step names
            execution_queue: Queue for execution
            packages: Required packages

        Returns:
            Self for chaining
        """
        if not self.pipeline:
            self.create_pipeline()

        self.pipeline.add_function_step(
            name=name,
            function=function,
            function_kwargs=function_kwargs or {},
            parents=parents or [],
            execution_queue=execution_queue or self.default_queue,
            packages=packages,
        )

        logger.info(f"Added function step: {name}")
        return self

    def start(
        self,
        queue: Optional[str] = None,
        wait: bool = True,
    ) -> str:
        """
        Start pipeline execution.

        Args:
            queue: Execution queue
            wait: Wait for completion

        Returns:
            Pipeline run ID
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not created")

        self.pipeline.start(queue=queue or self.default_queue)

        if wait:
            self.pipeline.wait()

        run_id = self.pipeline.pipeline_task.id
        logger.info(f"Pipeline started: {run_id}")
        return run_id

    def start_locally(self, run_pipeline_steps_locally: bool = True) -> str:
        """
        Start pipeline locally for debugging.

        Args:
            run_pipeline_steps_locally: Run steps locally

        Returns:
            Pipeline run ID
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not created")

        self.pipeline.start_locally(
            run_pipeline_steps_locally=run_pipeline_steps_locally
        )

        return self.pipeline.pipeline_task.id

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline execution status."""
        if not self.pipeline:
            return {"status": "not_started"}

        return {
            "status": self.pipeline.get_pipeline_execution_status(),
            "steps": {
                step_name: self.pipeline.get_step_status(step_name)
                for step_name in self.pipeline._steps
            },
        }


def create_ml_pipeline(
    project_name: str = "House Prices Prediction",
    pipeline_name: str = "HousePricesPipeline",
    default_queue: str = "default",
) -> ClearMLPipeline:
    """
    Create the standard ML pipeline.

    Returns:
        Configured ClearMLPipeline instance
    """
    pipeline = ClearMLPipeline(
        name=pipeline_name,
        project=project_name,
        default_queue=default_queue,
    )

    return pipeline


# Pipeline using decorators (alternative approach)
@PipelineDecorator.component(
    return_values=["data_stats"],
    cache=True,
)
def data_prepare_step(config: Dict[str, Any]) -> Dict[str, Any]:
    """Data preparation step."""
    import json
    import pandas as pd
    from pathlib import Path

    Path("data/interim").mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config["data"]["train_path"])
    train_df = train_df.drop_duplicates()

    stats = {
        "train_rows": len(train_df),
        "train_cols": len(train_df.columns),
    }

    train_df.to_csv("data/interim/train_prepared.csv", index=False)

    return stats


@PipelineDecorator.component(
    return_values=["feature_stats"],
    cache=True,
)
def feature_engineering_step(
    config: Dict[str, Any],
    data_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Feature engineering step."""
    import joblib
    import pandas as pd
    from pathlib import Path
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv("data/interim/train_prepared.csv")

    target_col = config["data"]["target_col"]
    y = train_df[target_col].values
    X = train_df.drop([target_col, "Id"], axis=1, errors="ignore")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    # Encode and scale
    X_train_encoded = pd.get_dummies(X_train, drop_first=True, dtype=float)
    X_val_encoded = pd.get_dummies(X_val, drop_first=True, dtype=float)
    X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train_encoded)
    X_val_imputed = imputer.transform(X_val_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    # Save
    pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns).to_csv(
        "data/processed/X_train.csv", index=False
    )
    pd.DataFrame(X_val_scaled, columns=X_train_encoded.columns).to_csv(
        "data/processed/X_val.csv", index=False
    )
    pd.DataFrame(y_train, columns=[target_col]).to_csv(
        "data/processed/y_train.csv", index=False
    )
    pd.DataFrame(y_val, columns=[target_col]).to_csv(
        "data/processed/y_val.csv", index=False
    )
    joblib.dump(scaler, "models/scaler.joblib")

    return {
        "n_features": X_train_encoded.shape[1],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }


@PipelineDecorator.component(
    return_values=["train_metrics", "model_path"],
    cache=False,
)
def train_step(
    config: Dict[str, Any],
    feature_stats: Dict[str, Any],
) -> tuple:
    """Model training step."""
    import joblib
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_train = pd.read_csv("data/processed/X_train.csv")
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    model_cfg = config["model"]
    model_type = model_cfg["type"]
    model_params = {k: v for k, v in model_cfg.items() if k not in ["type", "_validate"]}

    if model_type == "RandomForestRegressor":
        model = RandomForestRegressor(**model_params)
    elif model_type == "GradientBoostingRegressor":
        model = GradientBoostingRegressor(**model_params)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    metrics = {
        "train_mse": float(mean_squared_error(y_train, y_pred_train)),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "val_mse": float(mean_squared_error(y_val, y_pred_val)),
        "val_r2": float(r2_score(y_val, y_pred_val)),
        "val_mae": float(mean_absolute_error(y_val, y_pred_val)),
    }

    model_path = "models/model.joblib"
    joblib.dump(model, model_path)

    return metrics, model_path


@PipelineDecorator.component(
    return_values=["eval_metrics"],
    cache=False,
)
def evaluate_step(
    config: Dict[str, Any],
    train_metrics: Dict[str, float],
    model_path: str,
) -> Dict[str, float]:
    """Model evaluation step."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model = joblib.load(model_path)
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    y_pred = model.predict(X_val)

    metrics = {
        "eval_mse": float(mean_squared_error(y_val, y_pred)),
        "eval_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "eval_mae": float(mean_absolute_error(y_val, y_pred)),
        "eval_r2": float(r2_score(y_val, y_pred)),
    }

    return metrics


@PipelineDecorator.pipeline(
    name="HousePricesPipeline",
    project="House Prices Prediction",
    version="1.0.0",
)
def house_prices_pipeline(config: Dict[str, Any]):
    """Complete ML pipeline for house prices prediction."""
    data_stats = data_prepare_step(config)
    feature_stats = feature_engineering_step(config, data_stats)
    train_metrics, model_path = train_step(config, feature_stats)
    eval_metrics = evaluate_step(config, train_metrics, model_path)

    print(f"Pipeline completed. Final metrics: {eval_metrics}")
    return eval_metrics


class PipelineScheduler:
    """Schedule and trigger ClearML pipelines."""

    def __init__(self, project_name: str = "House Prices Prediction"):
        """Initialize scheduler."""
        self.project_name = project_name
        self.scheduler: Optional[TriggerScheduler] = None

    def create_schedule(
        self,
        pipeline_task_id: str,
        schedule_cron: str = "0 0 * * *",  # Daily at midnight
        name: str = "DailyPipelineRun",
    ) -> TriggerScheduler:
        """
        Create a scheduled pipeline trigger.

        Args:
            pipeline_task_id: ID of the pipeline task to schedule
            schedule_cron: Cron expression for schedule
            name: Name for the scheduler

        Returns:
            TriggerScheduler instance
        """
        self.scheduler = TriggerScheduler(
            project=self.project_name,
            name=name,
        )

        # Add scheduled task
        self.scheduler.add_task_trigger(
            task_id=pipeline_task_id,
            target_project=self.project_name,
            trigger_schedule=schedule_cron,
        )

        return self.scheduler

    def start(self) -> None:
        """Start the scheduler."""
        if self.scheduler:
            self.scheduler.start()
            logger.info("Pipeline scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler:
            self.scheduler.stop()
            logger.info("Pipeline scheduler stopped")


def create_pipeline() -> ClearMLPipeline:
    """Factory function to create the standard pipeline."""
    return create_ml_pipeline()