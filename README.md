Repository of "Engineering practices in Machine Learning" course

## Documentation
- [HW Report 1](./reports/REPORT1.md) - Project setup and structure
- [HW Report 2](./reports/REPORT2.md) - Data and Model Versioning Setup
- [HW Report 3](./reports/REPORT3.md) - Experiment tracking with MLflow
- [HW Report 4](./reports/REPORT4.md) - ML Pipeline Automation (DVC + Hydra)


## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Poetry
- Git
- DVC 3.x

### Installation

```bash
# Clone repository
git clone https://github.com/alllyuk/itmo-epml
cd itmo-epml

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Install pre-commit hooks
pre-commit install

# Pull DVC data and cache
dvc pull
```

### Running ML Pipeline
```bash
# Run full training pipeline with MLflow tracking
poetry run python src/itmo_epml/main.py

# View MLflow dashboard (opens at http://localhost:5000)
poetry run mlflow ui --backend-store-uri file:///$(pwd)/mlruns

# List model versions and compare
poetry run python src/itmo_epml/model_registry.py
```

### Data and Model Versioning
```bash
# Initialize/pull data versions
poetry run dvc pull

# Add new data to versioning
poetry run dvc add data/raw/yourfile.csv

# Push data versions to remote storage
poetry run dvc push

# View data version history
poetry run dvc dag

# Run full DVC pipeline
dvc repro
```

### Hydra Configuration
```bash
# Default run
python -m src.itmo_epml.run_pipeline

# Specific model
python -m src.itmo_epml.run_pipeline model=gradient_boosting

# Specific experiment
python -m src.itmo_epml.run_pipeline experiment=optimized

# Grid Search (multi-run)
python -m src.itmo_epml.stages.train --multirun \
    model.n_estimators=50,100,200 \
    model.max_depth=5,10,20
```

### Running Tests
```bash
poetry run pytest
```

### Code Quality
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Check security
poetry run bandit -r src/ -c pyproject.toml
```

### Project Structure
```
itmo-epml/
├── configs/
│ ├── config.yaml # Main Hydra config
│ ├── data/ # Data configs
│ ├── model/ # Model configs (RF, LR, GB)
│ ├── training/ # Training configs
│ ├── mlflow/ # MLflow configs
│ └── experiment/ # Experiment presets
├── data/
│ ├── external/
│ ├── raw/ # Original data
│ ├── interim/ # Prepared data (DVC cached)
│ └── processed/ # Final features (DVC cached)
├── models/ # Trained models and transformers
├── reports/
│ ├── figures/ # Plots
│ ├── monitoring/ # Pipeline execution reports
│ └── notifications/ # Notification logs
├── src/
│ └── itmo_epml/
│ ├── stages/ # DVC pipeline stages
│ │ ├── data_prepare.py
│ │ ├── feature_engineering.py
│ │ ├── train.py
│ │ └── evaluate.py
│ ├── main.py
│ ├── model_registry.py
│ ├── monitoring.py
│ └── notifications.py
├── tests/
├── dvc.yaml # DVC pipeline definition
├── params.yaml # DVC parameters
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

### Docker
```bash
# Build image
docker build -t itmo_epml .
# Run container
docker run -it itmo_epml
```

## Branching strategies

| Branch        | Purpose            |
|--------------|---------------------|
| main         | Stable version    |
| develop      | Features integration       |
| feature/*    | New features     |
| hotfix/*     | Urgent fixes  |
| hwN     | Homework humber N  |

## Author

Alexey Kornelyuk
