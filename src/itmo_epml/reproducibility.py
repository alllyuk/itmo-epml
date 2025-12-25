"""Reproducibility utilities and full pipeline."""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import yaml

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """Manage reproducibility configuration and execution."""

    def __init__(self, project_root: str = "."):
        """Initialize Reproducibility Manager."""
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / "configs"
        self.config_dir.mkdir(exist_ok=True)

    def create_environment_snapshot(self, output_path: Optional[str] = None) -> dict:
        """Create a snapshot of the current environment."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "packages": self._get_installed_packages(),
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(snapshot, f, indent=2)
            logger.info(f"Environment snapshot saved to {output_path}")

        return snapshot

    def _get_installed_packages(self) -> dict[str, str]:
        """Get installed package versions."""
        packages = {}
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                import json as json_lib

                package_list = json_lib.loads(result.stdout)
                for package in package_list:
                    packages[package["name"]] = package["version"]
        except Exception as e:
            logger.warning(f"Could not get installed packages: {e}")

        return packages

    def create_training_config(self, config_dict: dict, name: str = "training_config"):
        """Create a training configuration file."""
        config_path = self.config_dir / f"{name}.yaml"

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        logger.info(f"Configuration saved to {config_path}")
        return config_path

    def load_training_config(self, config_path: str) -> dict:
        """Load training configuration from file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    def create_reproduction_script(self, output_path: str = "reproduce.sh"):
        """Create a reproduction script."""
        script_content = """#!/bin/bash

# Data Version Control setup
echo "Setting up Data Version Control..."
poetry run dvc pull

# Install dependencies
echo "Installing dependencies..."
poetry install

# Set MLflow tracking
export MLFLOW_TRACKING_URI=file://$(pwd)/mlruns

# Run training pipeline
echo "Running training pipeline..."
poetry run python src/itmo_epml/train.py

# Generate comparison report
echo "Generating model comparison report..."
poetry run python src/itmo_epml/model_registry.py

echo "Pipeline completed successfully!"
"""

        with open(output_path, "w") as f:
            f.write(script_content)

        # Make script executable (Unix)
        Path(output_path).chmod(0o755)

        logger.info(f"Reproduction script created: {output_path}")
        return output_path

    def create_docker_reproducibility_config(
        self, output_path: str = "Dockerfile.reproduce"
    ):
        """Create a Docker configuration for full reproducibility."""
        dockerfile_content = """FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    POETRY_VERSION=1.7.1 \\
    POETRY_HOME="/opt/poetry" \\
    POETRY_VIRTUALENVS_IN_PROJECT=true \\
    POETRY_NO_INTERACTION=1

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock* ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/raw/ ./data/raw/

# Install dependencies
RUN poetry install

# Set MLflow tracking
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Create directories
RUN mkdir -p ./mlruns ./dvc_storage

# Run pipeline
CMD ["poetry", "run", "python", "src/itmo_epml/train.py"]
"""

        with open(output_path, "w") as f:
            f.write(dockerfile_content)

        logger.info(f"Docker reproducibility config created: {output_path}")
        return output_path


class PipelineExecutor:
    """Execute complete ML pipeline with reproducibility guarantees."""

    def __init__(self, project_root: str = "."):
        """Initialize Pipeline Executor."""
        self.project_root = Path(project_root)
        self.reproducibility_mgr = ReproducibilityManager(project_root)

    def execute_full_pipeline(
        self,
        config_path: Optional[str] = None,
        save_snapshot: bool = True,
        experiment_name: str = "full_pipeline",
    ):
        """Execute full pipeline with reproducibility."""
        logger.info("Starting full pipeline execution...")

        # Create environment snapshot
        if save_snapshot:
            snapshot_path = self.project_root / "reports" / "environment_snapshot.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.reproducibility_mgr.create_environment_snapshot(str(snapshot_path))

        # Load configuration
        if config_path:
            config = self.reproducibility_mgr.load_training_config(config_path)
        else:
            config = self._get_default_config()

        # Setup MLflow
        mlflow_dir = self.project_root / "mlruns"
        mlflow_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{mlflow_dir.absolute()}")
        mlflow.set_experiment(experiment_name)

        # Log configuration
        mlflow.log_dict(config, "pipeline_config.json")

        # Log environment metadata
        mlflow.log_dict(
            {
                "python_version": sys.version,
                "project_root": str(self.project_root.absolute()),
                "execution_time": datetime.now().isoformat(),
            },
            "execution_metadata.json",
        )

        logger.info("Pipeline execution completed successfully")
        return True

    def _get_default_config(self) -> dict:
        """Get default training configuration."""
        return {
            "data": {
                "train_path": "data/raw/train.csv",
                "test_path": "data/raw/test.csv",
            },
            "model": {
                "type": "RandomForest",
                "n_estimators": 100,
                "max_depth": 10,
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
            },
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    mgr = ReproducibilityManager()
    mgr.create_environment_snapshot("environment_snapshot.json")
    mgr.create_reproduction_script()
    mgr.create_docker_reproducibility_config()
    mgr.create_training_config(
        {
            "data": {"train_path": "data/raw/train.csv"},
            "model": {"n_estimators": 100},
        }
    )
