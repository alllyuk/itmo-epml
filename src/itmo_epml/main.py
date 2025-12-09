"""Main entry point for itmo-epml."""

import logging
from pathlib import Path

from itmo_epml.reproducibility import PipelineExecutor, ReproducibilityManager
from itmo_epml.train import main as train_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main function - execute full ML pipeline."""
    logger.info("Starting itmo-epml ML pipeline...")

    project_root = Path(__file__).parent.parent.parent

    # Create reproducibility configuration
    logger.info("Setting up reproducibility environment...")
    repro_mgr = ReproducibilityManager(str(project_root))
    repro_mgr.create_environment_snapshot(
        str(project_root / "reports" / "environment_snapshot.json")
    )

    # Execute training pipeline
    logger.info("Training model...")
    model, metrics, run_id = train_main()

    logger.info(f"Model training completed. Run ID: {run_id}")
    logger.info(f"Metrics: {metrics}")

    # Execute full pipeline with reproducibility
    logger.info("Finalizing pipeline execution...")
    executor = PipelineExecutor(str(project_root))
    executor.execute_full_pipeline(
        config_path=str(project_root / "configs" / "training_config.yaml"),
        save_snapshot=True,
    )

    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
