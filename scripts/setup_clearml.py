#!/usr/bin/env python3
"""Setup script for ClearML initialization."""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_clearml_server():
    """Check if ClearML server is accessible."""
    import requests

    servers = {
        "API Server": "http://localhost:8008",
        "Web Server": "http://localhost:8080",
        "File Server": "http://localhost:8081",
    }

    all_ok = True
    for name, url in servers.items():
        try:
            response = requests.get(f"{url}/", timeout=5)
            status = "✓ OK" if response.status_code < 500 else f"✗ Error ({response.status_code})"
        except requests.exceptions.ConnectionError:
            status = "✗ Not accessible"
            all_ok = False
        except Exception as e:
            status = f"✗ Error: {e}"
            all_ok = False

        logger.info(f"{name}: {status}")

    return all_ok


def setup_clearml_credentials():
    """Setup ClearML credentials interactively or from environment."""
    from clearml import Task

    # Check for existing credentials
    api_key = os.getenv("CLEARML_API_ACCESS_KEY")
    secret_key = os.getenv("CLEARML_API_SECRET_KEY")

    if api_key and secret_key:
        logger.info("Using credentials from environment variables")
        os.environ["CLEARML_API_HOST"] = "http://localhost:8008"
        os.environ["CLEARML_WEB_HOST"] = "http://localhost:8080"
        os.environ["CLEARML_FILES_HOST"] = "http://localhost:8081"
        return True

    # Try to initialize without credentials (open server)
    logger.info("Attempting connection to open ClearML server...")
    os.environ["CLEARML_API_HOST"] = "http://localhost:8008"
    os.environ["CLEARML_WEB_HOST"] = "http://localhost:8080"
    os.environ["CLEARML_FILES_HOST"] = "http://localhost:8081"

    try:
        # Test connection
        task = Task.init(
            project_name="Test",
            task_name="connection_test",
            reuse_last_task_id=False,
        )
        task.close()
        logger.info("✓ Successfully connected to ClearML server")
        return True
    except Exception as e:
        logger.warning(f"Connection test failed: {e}")
        logger.info("Run 'clearml-init' to configure credentials")
        return False


def create_project():
    """Create the main project in ClearML."""
    from clearml import Task

    try:
        task = Task.init(
            project_name="House Prices Prediction",
            task_name="project_initialization",
            task_type=Task.TaskTypes.custom,
            reuse_last_task_id=False,
        )

        task.set_comment("Project initialized for House Prices ML Pipeline")
        task.add_tags(["initialization", "setup"])

        logger.info("✓ Project 'House Prices Prediction' created")
        task.close()
        return True
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        return False


def run_test_experiment():
    """Run a simple test experiment."""
    from clearml import Task

    try:
        task = Task.init(
            project_name="House Prices Prediction",
            task_name="test_experiment",
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,
        )

        # Log some test parameters
        task.connect({
            "test_param_1": 100,
            "test_param_2": "value",
            "learning_rate": 0.01,
        })

        # Log test metrics
        logger_task = task.get_logger()
        for i in range(10):
            logger_task.report_scalar("test", "loss", value=1.0/(i+1), iteration=i)
            logger_task.report_scalar("test", "accuracy", value=i*0.1, iteration=i)

        task.add_tags(["test", "validation"])
        logger.info(f"✓ Test experiment completed: {task.id}")
        logger.info(f"  View at: {task.get_output_log_web_page()}")

        task.close()
        return True
    except Exception as e:
        logger.error(f"Test experiment failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("ClearML Setup Script")
    print("=" * 60)

    # Step 1: Check server
    print("\n1. Checking ClearML Server...")
    if not check_clearml_server():
        print("\n⚠️  ClearML server is not fully accessible.")
        print("   Start the server with: docker-compose up -d")
        print("   Wait a few minutes for services to initialize.")
        return False

    # Step 2: Setup credentials
    print("\n2. Setting up credentials...")
    if not setup_clearml_credentials():
        print("\n⚠️  Could not setup credentials.")
        return False

    # Step 3: Create project
    print("\n3. Creating project...")
    if not create_project():
        print("\n⚠️  Could not create project.")
        return False

    # Step 4: Run test experiment
    print("\n4. Running test experiment...")
    if not run_test_experiment():
        print("\n⚠️  Test experiment failed.")
        return False

    print("\n" + "=" * 60)
    print("✓ ClearML setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Open ClearML Web UI: http://localhost:8080")
    print("2. Run pipeline: python scripts/run_clearml_pipeline.py")
    print("3. View experiments in the dashboard")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)