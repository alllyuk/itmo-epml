"""ClearML configuration and initialization."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from clearml import Task

logger = logging.getLogger(__name__)


@dataclass
class ClearMLConfig:
    """ClearML configuration container."""

    web_server: str = "http://localhost:8080"
    api_server: str = "http://localhost:8008"
    files_server: str = "http://localhost:8081"
    project_name: str = "House Prices Prediction"
    task_name: Optional[str] = None
    task_type: str = "training"
    auto_connect_frameworks: bool = True
    auto_connect_arg_parser: bool = True
    tags: list = field(default_factory=list)
    reuse_last_task_id: bool = False

    @classmethod
    def from_hydra(cls, cfg) -> "ClearMLConfig":
        """Create config from Hydra DictConfig."""
        return cls(
            web_server=cfg.clearml.server.web_server,
            api_server=cfg.clearml.server.api_server,
            files_server=cfg.clearml.server.files_server,
            project_name=cfg.clearml.project.name,
            task_type=cfg.clearml.task.task_type,
            auto_connect_frameworks=cfg.clearml.task.auto_connect_frameworks,
            auto_connect_arg_parser=cfg.clearml.task.auto_connect_arg_parser,
        )


def setup_clearml_credentials(
    api_server: str,
    web_server: str,
    files_server: str,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> None:
    """Setup ClearML credentials programmatically."""
    # Установка через переменные окружения
    os.environ["CLEARML_API_HOST"] = api_server
    os.environ["CLEARML_WEB_HOST"] = web_server
    os.environ["CLEARML_FILES_HOST"] = files_server

    if access_key and secret_key:
        os.environ["CLEARML_API_ACCESS_KEY"] = access_key
        os.environ["CLEARML_API_SECRET_KEY"] = secret_key

    logger.info(f"ClearML configured: API={api_server}, Web={web_server}")


def init_clearml(
    config: ClearMLConfig,
    task_name: Optional[str] = None,
    continue_last_task: bool = False,
) -> Task:
    """
    Initialize ClearML Task.

    Args:
        config: ClearML configuration
        task_name: Name for the task
        continue_last_task: Whether to continue last task with same name

    Returns:
        ClearML Task instance
    """
    setup_clearml_credentials(
        api_server=config.api_server,
        web_server=config.web_server,
        files_server=config.files_server,
    )

    task_name = task_name or config.task_name or "default_task"

    task = Task.init(
        project_name=config.project_name,
        task_name=task_name,
        task_type=getattr(Task.TaskTypes, config.task_type, Task.TaskTypes.training),
        auto_connect_frameworks=config.auto_connect_frameworks,
        auto_connect_arg_parser=config.auto_connect_arg_parser,
        reuse_last_task_id=config.reuse_last_task_id or continue_last_task,
        tags=config.tags if config.tags else None,
    )

    logger.info(f"ClearML Task initialized: {task.id}")
    return task


def get_clearml_task() -> Optional[Task]:
    """Get current ClearML task if exists."""
    return Task.current_task()


def close_clearml_task(task: Optional[Task] = None) -> None:
    """Close ClearML task."""
    task = task or Task.current_task()
    if task:
        task.close()
        logger.info(f"ClearML Task {task.id} closed")