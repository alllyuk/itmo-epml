# Конфигурация

Проект использует Hydra для управления конфигурациями.

## Структура конфигураций

```
configs/
├── config.yaml          # Главный конфиг с defaults
├── data/
│   ├── default.yaml     # Параметры данных по умолчанию
│   └── house_prices.yaml
├── model/
│   ├── random_forest.yaml
│   └── gradient_boosting.yaml
├── training/
│   └── default.yaml
├── mlflow/
│   └── default.yaml
├── clearml/
│   └── default.yaml
└── experiment/
    ├── baseline.yaml
    └── optimized.yaml
```

## Главный конфиг

```yaml title="configs/config.yaml"
defaults:
  - data: default
  - model: gradient_boosting
  - training: default
  - mlflow: default
  - clearml: default
  - _self_

project_name: "House Prices Prediction"
random_seed: 42
```

## Конфигурация данных

```yaml title="configs/data/default.yaml"
raw_train_path: "data/raw/train.csv"
raw_test_path: "data/raw/test.csv"
interim_train_path: "data/interim/train_prepared.csv"
interim_test_path: "data/interim/test_prepared.csv"
processed_dir: "data/processed"
target_col: "SalePrice"
test_size: 0.2
random_state: 42
```

## Конфигурация модели

=== "Gradient Boosting"

    ```yaml title="configs/model/gradient_boosting.yaml"
    name: "GradientBoostingRegressor"
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
    min_samples_split: 2
    min_samples_leaf: 1
    random_state: ${random_seed}
    ```

=== "Random Forest"

    ```yaml title="configs/model/random_forest.yaml"
    name: "RandomForestRegressor"
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    min_samples_leaf: 1
    random_state: ${random_seed}
    n_jobs: -1
    ```

## Конфигурация MLflow

```yaml title="configs/mlflow/default.yaml"
tracking_uri: "sqlite:///mlflow.db"
experiment_name: "House Prices DVC Pipeline"
model_name: "HousePriceModel"
log_feature_importance: true
register_model: true
min_r2_for_registration: 0.5
```

## Пресеты экспериментов

=== "Baseline"

    ```yaml title="configs/experiment/baseline.yaml"
    # @package _global_
    defaults:
      - override /model: random_forest

    model:
      n_estimators: 100
      max_depth: 10

    experiment_name: "baseline"
    ```

=== "Optimized"

    ```yaml title="configs/experiment/optimized.yaml"
    # @package _global_
    defaults:
      - override /model: random_forest

    model:
      n_estimators: 200
      max_depth: 20

    training:
      cross_validation: true
      cv_folds: 5

    experiment_name: "optimized"
    ```

## Переопределение параметров

### Из командной строки

```bash
# Один параметр
python -m src.itmo_epml.run_pipeline model.n_estimators=200

# Несколько параметров
python -m src.itmo_epml.run_pipeline \
    model.n_estimators=200 \
    model.max_depth=15 \
    training.cross_validation=true
```

### Выбор конфига

```bash
# Использовать random_forest вместо gradient_boosting
python -m src.itmo_epml.run_pipeline model=random_forest

# Использовать пресет эксперимента
python -m src.itmo_epml.run_pipeline experiment=optimized
```

### Multi-run (Grid Search)

```bash
python -m src.itmo_epml.stages.train --multirun \
    model.n_estimators=50,100,200 \
    model.max_depth=5,10,20 \
    model.min_samples_split=2,5
```

## Переменные окружения

Создайте файл `.env` на основе `.env.example`:

```bash title=".env"
# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# ClearML
CLEARML_API_ACCESS_KEY=your_access_key
CLEARML_API_SECRET_KEY=your_secret_key
CLEARML_API_HOST=http://localhost:8008
```

## Параметры DVC

```yaml title="params.yaml"
data:
  test_size: 0.2
  random_state: 42
  target_col: SalePrice

model:
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  min_samples_split: 2

training:
  cross_validation: false
  cv_folds: 5

mlflow:
  experiment_name: "House Prices DVC Pipeline"
  model_name: "HousePriceModel"
```

## Интерполяция значений

Hydra поддерживает интерполяцию:

```yaml
random_seed: 42

model:
  random_state: ${random_seed}  # Использует значение random_seed

paths:
  data_dir: "data"
  raw_dir: "${paths.data_dir}/raw"  # data/raw
```
