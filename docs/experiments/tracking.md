# Трекинг экспериментов

Документация по настройке и использованию систем трекинга экспериментов.

## MLflow

MLflow используется для локального трекинга экспериментов.

### Запуск MLflow UI

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Доступ: [http://localhost:5000](http://localhost:5000)

### Конфигурация

```yaml title="configs/mlflow/default.yaml"
tracking_uri: "sqlite:///mlflow.db"
experiment_name: "House Prices DVC Pipeline"
model_name: "HousePriceModel"
log_feature_importance: true
register_model: true
min_r2_for_registration: 0.5
```

### Логирование в коде

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("House Prices DVC Pipeline")

with mlflow.start_run():
    # Параметры
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 5,
    })

    # Метрики
    mlflow.log_metrics({
        "val_r2": 0.909,
        "val_mse": 697213256.59,
    })

    # Модель
    mlflow.sklearn.log_model(model, "model")

    # Артефакты
    mlflow.log_artifact("reports/metrics/eval_metrics.json")
```

### Model Registry

```python
# Регистрация модели
mlflow.register_model(
    f"runs:/{run_id}/model",
    "HousePriceModel"
)

# Получение лучшей модели
from mlflow.tracking import MlflowClient
client = MlflowClient()
versions = client.search_model_versions("name='HousePriceModel'")
```

## ClearML

ClearML обеспечивает удалённое управление экспериментами.

### Запуск ClearML Server

```bash
docker-compose -f docker-compose.clearml.yml up -d
```

Доступ: [http://localhost:8080](http://localhost:8080)

### Инициализация

```bash
poetry run clearml-init
```

Или создайте `.env` файл:

```bash title=".env"
CLEARML_API_ACCESS_KEY=your_access_key
CLEARML_API_SECRET_KEY=your_secret_key
CLEARML_API_HOST=http://localhost:8008
```

### Логирование в коде

```python
from clearml import Task

task = Task.init(
    project_name="House Prices Prediction",
    task_name="experiment_1"
)

# Параметры
task.connect({"n_estimators": 100, "max_depth": 5})

# Метрики
logger = task.get_logger()
logger.report_scalar("validation", "r2", value=0.909, iteration=1)
logger.report_scalar("validation", "mse", value=697213256.59, iteration=1)
```

### Сравнение экспериментов

```python
from itmo_epml.clearml_integration import ExperimentComparator

comparator = ExperimentComparator("House Prices Prediction")
df = comparator.compare_metrics(["val_r2", "val_mse"])
best = comparator.get_best_experiment("val_r2", mode="max")
```

## Логируемые метрики

| Метрика | Тип | Описание |
|---------|-----|----------|
| `train_mse` | float | MSE на тренировочных данных |
| `train_rmse` | float | RMSE на тренировочных данных |
| `train_mae` | float | MAE на тренировочных данных |
| `train_r2` | float | R² на тренировочных данных |
| `val_mse` | float | MSE на валидационных данных |
| `val_rmse` | float | RMSE на валидационных данных |
| `val_mae` | float | MAE на валидационных данных |
| `val_r2` | float | R² на валидационных данных |
| `execution_time_seconds` | float | Время выполнения |

## Логируемые параметры

| Параметр | Тип | Описание |
|----------|-----|----------|
| `n_estimators` | int | Количество деревьев |
| `max_depth` | int | Максимальная глубина дерева |
| `learning_rate` | float | Скорость обучения |
| `min_samples_split` | int | Мин. сэмплов для split |
| `min_samples_leaf` | int | Мин. сэмплов в листе |
| `random_state` | int | Random seed |

## Артефакты

| Артефакт | Формат | Описание |
|----------|--------|----------|
| `model.joblib` | binary | Сериализованная модель |
| `feature_importance.json` | JSON | Важность признаков |
| `predictions_vs_actual.csv` | CSV | Предсказания vs факт |
| `residuals.csv` | CSV | Остатки модели |
| `eval_metrics.json` | JSON | Финальные метрики |

## Сравнение экспериментов

### MLflow

```bash
# Через UI
poetry run mlflow ui

# Через CLI
mlflow runs list --experiment-name "House Prices DVC Pipeline"
```

### ClearML

```bash
# Генерация отчёта
poetry run python scripts/compare_experiments.py --type all
```

### DVC

```bash
# Сравнение метрик
dvc metrics diff

# История метрик
dvc metrics show --all-commits
```

## Best Practices

1. **Именование экспериментов**: Используйте понятные имена с параметрами
   ```
   experiment_1_est100_depth5
   ```

2. **Версионирование**: Связывайте эксперименты с git commits

3. **Документация**: Добавляйте описания к экспериментам

4. **Теги**: Используйте теги для группировки
   ```python
   mlflow.set_tag("model_type", "gradient_boosting")
   ```

5. **Артефакты**: Сохраняйте все важные файлы
