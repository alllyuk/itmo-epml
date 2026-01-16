# Обучение модели

Стадия `train` обучает модель и логирует результаты в MLflow.

## Описание

Эта стадия выполняет:

1. Загрузку обработанных данных
2. Инициализацию модели с параметрами из конфига
3. Обучение модели
4. Логирование в MLflow
5. Вычисление метрик
6. Сохранение модели
7. Регистрацию в Model Registry

## Входные данные

| Файл | Описание |
|------|----------|
| `data/processed/X_train.csv` | Признаки для обучения |
| `data/processed/y_train.csv` | Целевая переменная (train) |
| `data/processed/X_val.csv` | Признаки для валидации |
| `data/processed/y_val.csv` | Целевая переменная (val) |

## Выходные данные

| Файл | Описание |
|------|----------|
| `models/model.joblib` | Обученная модель |
| `reports/metrics/train_metrics.json` | Метрики обучения |
| `reports/figures/feature_importance.json` | Важность признаков |

## Реализация

```python title="src/itmo_epml/stages/train.py"
@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Настройка MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # Загрузка данных
        X_train = pd.read_csv("data/processed/X_train.csv")
        y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
        X_val = pd.read_csv("data/processed/X_val.csv")
        y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

        # Инициализация модели
        model = GradientBoostingRegressor(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            learning_rate=cfg.model.learning_rate,
            min_samples_split=cfg.model.min_samples_split,
            random_state=cfg.model.random_state
        )

        # Логирование параметров
        mlflow.log_params({
            "n_estimators": cfg.model.n_estimators,
            "max_depth": cfg.model.max_depth,
            "learning_rate": cfg.model.learning_rate,
        })

        # Обучение
        model.fit(X_train, y_train)

        # Предсказания
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Метрики
        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "val_mse": mean_squared_error(y_val, y_val_pred),
            "val_r2": r2_score(y_val, y_val_pred),
            "val_mae": mean_absolute_error(y_val, y_val_pred),
        }

        # Логирование метрик
        mlflow.log_metrics(metrics)

        # Сохранение модели
        joblib.dump(model, "models/model.joblib")
        mlflow.sklearn.log_model(model, "model")

        # Регистрация модели
        if metrics["val_r2"] > cfg.mlflow.min_r2_for_registration:
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                cfg.mlflow.model_name
            )
```

## Конфигурация модели

=== "Gradient Boosting"

    ```yaml title="configs/model/gradient_boosting.yaml"
    name: "GradientBoostingRegressor"
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
    min_samples_split: 2
    ```

=== "Random Forest"

    ```yaml title="configs/model/random_forest.yaml"
    name: "RandomForestRegressor"
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    n_jobs: -1
    ```

## Запуск

```bash
# Через DVC
dvc repro train

# Напрямую
python -m src.itmo_epml.stages.train

# С другой моделью
python -m src.itmo_epml.stages.train model=random_forest

# Grid Search
python -m src.itmo_epml.stages.train --multirun \
    model.n_estimators=50,100,200 \
    model.max_depth=5,10,20
```

## Метрики

Пример `train_metrics.json`:

```json
{
  "train_mse": 11205610.90,
  "train_rmse": 3347.48,
  "train_mae": 2699.43,
  "train_r2": 0.9981,
  "val_mse": 697213256.59,
  "val_rmse": 26404.80,
  "val_mae": 15895.85,
  "val_r2": 0.9091
}
```

## MLflow интеграция

### Просмотр экспериментов

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Логируемые артефакты

- Параметры модели
- Метрики обучения и валидации
- Сериализованная модель
- Feature importance (JSON)
- Графики (при включении)

### Model Registry

Модели автоматически регистрируются при `val_r2 > 0.5`:

```python
mlflow.register_model(
    f"runs:/{run_id}/model",
    "HousePriceModel"
)
```

## Важность признаков

```json title="reports/figures/feature_importance.json"
{
  "OverallQual": 0.4521,
  "GrLivArea": 0.1234,
  "TotalBsmtSF": 0.0876,
  "GarageCars": 0.0654,
  "YearBuilt": 0.0543
}
```
