Repository of "Engineering practices in Machine Learning" course

## Documentation
[HW Report](./REPORT.md)


## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Poetry
- Git

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
├── configs/           # Configuration files (Hydra, etc.)
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned, transformed data
│   └── external/     # Data from external sources
├── models/           # Trained model files
├── notebooks/        # Jupyter notebooks for exploration
├── reports/
│   └── figures/      # Generated graphics and figures
├── src/
│   └── itmo_epml/  # Source code
├── tests/            # Unit tests
├── .pre-commit-config.yaml
├── pyproject.toml
├── .gitignore
├── Dockerfile
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
