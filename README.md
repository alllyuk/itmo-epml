# itmo-epml

Repository of "Engineering practices in Machine Learning" course

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Poetry
- Git

### Installation

```bash
# Clone repository
git clone <repository-url>
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
poetry run isort src/ tests/

# Lint code
poetry run ruff check src/ tests/
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

### Author

Alexey Kornelyuk
