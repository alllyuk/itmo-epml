"""Main script to run the complete pipeline with monitoring."""

import logging
import subprocess
import sys
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from src.itmo_epml.monitoring import PipelineMonitor, get_monitor
from src.itmo_epml.notifications import (
    ConsoleNotification,
    FileNotification,
    NotificationManager,
    PipelineResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_dvc_pipeline(
    stages: list[str] = None,
    force: bool = False,
) -> tuple[bool, str]:
    """Run DVC pipeline with optional stage selection."""

    cmd = ["dvc", "repro"]

    if stages:
        for stage in stages:
            cmd.extend(["--single-item", stage])

    if force:
        cmd.append("--force")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def update_params_from_hydra(cfg: DictConfig) -> None:
    """Update params.yaml with Hydra configuration."""
    params_path = Path("params.yaml")

    # Convert OmegaConf to dict
    params = OmegaConf.to_container(cfg, resolve=True)

    # Remove hydra-specific keys
    params.pop("hydra", None)

    with open(params_path, "w") as f:
        yaml.dump(params, f, default_flow_style=False)

    logger.info(f"Updated {params_path} with Hydra configuration")


def load_metrics() -> dict:
    """Load metrics from pipeline outputs."""
    metrics = {}

    metrics_files = [
        "reports/metrics/data_stats.json",
        "reports/metrics/train_metrics.json",
        "reports/metrics/eval_metrics.json",
    ]

    for file_path in metrics_files:
        try:
            with open(file_path) as f:
                file_metrics = json.load(f)
                metrics.update(file_metrics)
        except FileNotFoundError:
            logger.warning(f"Metrics file not found: {file_path}")

    return metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for pipeline execution."""

    logger.info("=" * 60)
    logger.info("Starting ML Pipeline with DVC + Hydra")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize monitoring
    monitor = PipelineMonitor(pipeline_name="HousePricesPipeline")

    # Initialize notifications
    notifier = (
        NotificationManager()
        .add_channel(ConsoleNotification())
        .add_channel(FileNotification())
    )

    try:
        # Update params.yaml with Hydra config
        monitor.start_stage("config_sync")
        update_params_from_hydra(cfg)
        monitor.end_stage("config_sync", status="success")

        # Run DVC pipeline
        monitor.start_stage("dvc_pipeline")
        success, output = run_dvc_pipeline()

        if not success:
            monitor.end_stage("dvc_pipeline", status="failed", error=output)
            raise RuntimeError(f"DVC pipeline failed: {output}")

        monitor.end_stage("dvc_pipeline", status="success")

        # Load and log metrics
        metrics = load_metrics()

        # Create result
        summary = monitor.get_summary()
        result = PipelineResult(
            pipeline_name=summary["pipeline_name"],
            status=summary["overall_status"],
            duration_seconds=summary["total_duration_seconds"],
            metrics=metrics,
            run_id=metrics.get("run_id"),
        )

        # Send notifications
        notifier.notify(result)

        # Save monitoring report
        monitor.save_report()
        monitor.print_summary()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

        result = PipelineResult(
            pipeline_name="HousePricesPipeline",
            status="failed",
            duration_seconds=0,
            metrics={},
            error=str(e),
        )
        notifier.notify(result)

        monitor.save_report()
        monitor.print_summary()

        sys.exit(1)


if __name__ == "__main__":
    import json  # Import here for load_metrics
    main()