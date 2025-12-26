"""ClearML integration module for ML experiment tracking and pipeline management."""

from src.itmo_epml.clearml_integration.config import (
    ClearMLConfig,
    init_clearml,
    get_clearml_task,
)
from src.itmo_epml.clearml_integration.experiment import (
    ExperimentTracker,
    create_experiment,
    log_metrics,
    log_artifacts,
)
from src.itmo_epml.clearml_integration.model_registry import (
    ClearMLModelRegistry,
    register_model,
    compare_models,
)
from src.itmo_epml.clearml_integration.pipeline import (
    ClearMLPipeline,
    create_pipeline,
)

__all__ = [
    "ClearMLConfig",
    "init_clearml",
    "get_clearml_task",
    "ExperimentTracker",
    "create_experiment",
    "log_metrics",
    "log_artifacts",
    "ClearMLModelRegistry",
    "register_model",
    "compare_models",
    "ClearMLPipeline",
    "create_pipeline",
]