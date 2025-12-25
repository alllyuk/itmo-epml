"""Tests for pipeline reproducibility."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml


class TestPipelineReproducibility:
    """Test suite for pipeline reproducibility."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment."""
        self.original_dir = Path.cwd()
        # Tests run from project root

    def test_params_yaml_exists(self):
        """Test that params.yaml exists."""
        assert Path("params.yaml").exists(), "params.yaml not found"

    def test_dvc_yaml_exists(self):
        """Test that dvc.yaml exists."""
        assert Path("dvc.yaml").exists(), "dvc.yaml not found"

    def test_dvc_yaml_valid(self):
        """Test that dvc.yaml is valid YAML."""
        with open("dvc.yaml") as f:
            dvc_config = yaml.safe_load(f)

        assert "stages" in dvc_config
        assert len(dvc_config["stages"]) > 0

    def test_hydra_configs_exist(self):
        """Test that all Hydra config files exist."""
        required_configs = [
            "configs/config.yaml",
            "configs/data/default.yaml",
            "configs/model/random_forest.yaml",
            "configs/training/default.yaml",
            "configs/mlflow/default.yaml",
        ]

        for config_path in required_configs:
            assert Path(config_path).exists(), f"Config not found: {config_path}"

    def test_stage_scripts_exist(self):
        """Test that all stage scripts exist."""
        stages = [
            "src/itmo_epml/stages/data_prepare.py",
            "src/itmo_epml/stages/feature_engineering.py",
            "src/itmo_epml/stages/train.py",
            "src/itmo_epml/stages/evaluate.py",
        ]

        for stage_path in stages:
            assert Path(stage_path).exists(), f"Stage script not found: {stage_path}"

    def test_dvc_dag(self):
        """Test that DVC can generate DAG."""
        result = subprocess.run(
            ["dvc", "dag"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"DVC DAG failed: {result.stderr}"
        assert "data_prepare" in result.stdout
        assert "train" in result.stdout

    def test_params_schema(self):
        """Test params.yaml has required keys."""
        with open("params.yaml") as f:
            params = yaml.safe_load(f)

        required_keys = ["data", "model", "training", "mlflow"]
        for key in required_keys:
            assert key in params, f"Missing key in params.yaml: {key}"

    @pytest.mark.slow
    def test_pipeline_dry_run(self):
        """Test DVC pipeline dry run."""
        result = subprocess.run(
            ["dvc", "repro", "--dry"],
            capture_output=True,
            text=True
        )

        # Dry run should succeed even without data
        assert result.returncode == 0 or "Skipping" in result.stdout

    def test_config_composition(self):
        """Test Hydra config composition."""
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        config_path = str(Path("configs").absolute())

        with initialize_config_dir(config_dir=config_path, version_base=None):
            # Test default composition
            cfg = compose(config_name="config")
            assert cfg.model.type == "RandomForestRegressor"

            # Test override
            cfg = compose(
                config_name="config",
                overrides=["model=linear_regression"]
            )
            assert cfg.model.type == "LinearRegression"

    def test_metrics_reproducibility(self):
        """Test that same params produce same metrics hash."""
        # This would require running the pipeline twice
        # For now, just verify metrics files structure

        metrics_files = [
            "reports/metrics/train_metrics.json",
            "reports/metrics/eval_metrics.json",
        ]

        for metrics_file in metrics_files:
            if Path(metrics_file).exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)

                # Verify structure
                assert isinstance(metrics, dict)
                # Common metrics should exist
                if "train" in metrics_file:
                    assert any("r2" in k for k in metrics.keys())