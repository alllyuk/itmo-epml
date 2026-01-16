# Оценка модели

Стадия `evaluate` оценивает обученную модель на валидационных данных.

## Описание

Эта стадия выполняет:

1. Загрузку обученной модели
2. Предсказание на валидационных данных
3. Вычисление финальных метрик
4. Анализ остатков
5. Создание визуализаций
6. Логирование в MLflow

## Входные данные

| Файл | Описание |
|------|----------|
| `models/model.joblib` | Обученная модель |
| `data/processed/X_val.csv` | Валидационные признаки |
| `data/processed/y_val.csv` | Валидационная целевая переменная |

## Выходные данные

| Файл | Описание |
|------|----------|
| `reports/metrics/eval_metrics.json` | Финальные метрики |
| `reports/figures/predictions_vs_actual.csv` | Предсказания vs факт |
| `reports/figures/residuals.csv` | Остатки модели |

## Реализация

```python title="src/itmo_epml/stages/evaluate.py"
@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Загрузка модели и данных
    model = joblib.load("models/model.joblib")
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()

    # Предсказания
    y_pred = model.predict(X_val)

    # Метрики
    metrics = {
        "eval_mse": mean_squared_error(y_val, y_pred),
        "eval_rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
        "eval_mae": mean_absolute_error(y_val, y_pred),
        "eval_r2": r2_score(y_val, y_pred),
        "eval_mape": np.mean(np.abs((y_val - y_pred) / y_val)) * 100,
    }

    # Анализ остатков
    residuals = y_val - y_pred
    metrics["residual_mean"] = float(np.mean(residuals))
    metrics["residual_std"] = float(np.std(residuals))

    # Сохранение метрик
    with open("reports/metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions vs Actual
    pd.DataFrame({
        "actual": y_val,
        "predicted": y_pred
    }).to_csv("reports/figures/predictions_vs_actual.csv", index=False)

    # Residuals
    pd.DataFrame({
        "predicted": y_pred,
        "residual": residuals
    }).to_csv("reports/figures/residuals.csv", index=False)

    # MLflow логирование
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("reports/metrics/eval_metrics.json")
```

## Запуск

```bash
# Через DVC
dvc repro evaluate

# Напрямую
python -m src.itmo_epml.stages.evaluate
```

## Метрики

### Основные метрики

| Метрика | Описание | Формула |
|---------|----------|---------|
| MSE | Mean Squared Error | $\frac{1}{n}\sum(y - \hat{y})^2$ |
| RMSE | Root Mean Squared Error | $\sqrt{MSE}$ |
| MAE | Mean Absolute Error | $\frac{1}{n}\sum|y - \hat{y}|$ |
| R² | Coefficient of Determination | $1 - \frac{SS_{res}}{SS_{tot}}$ |
| MAPE | Mean Absolute Percentage Error | $\frac{100}{n}\sum|\frac{y - \hat{y}}{y}|$ |

### Пример выходных метрик

```json title="reports/metrics/eval_metrics.json"
{
  "eval_mse": 712662125.58,
  "eval_rmse": 26695.73,
  "eval_mae": 16060.17,
  "eval_r2": 0.9071,
  "eval_mape": 9.75,
  "residual_mean": 511.36,
  "residual_std": 26690.83
}
```

## Визуализации

### Predictions vs Actual

```csv title="reports/figures/predictions_vs_actual.csv"
actual,predicted
208500,215234.5
181500,178923.1
223500,220156.8
...
```

Используется для построения scatter plot:

```python
import matplotlib.pyplot as plt

df = pd.read_csv("reports/figures/predictions_vs_actual.csv")
plt.scatter(df["actual"], df["predicted"], alpha=0.5)
plt.plot([0, 500000], [0, 500000], "r--")  # идеальная линия
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
```

### Residuals

```csv title="reports/figures/residuals.csv"
predicted,residual
215234.5,-6734.5
178923.1,2576.9
220156.8,3343.2
...
```

Используется для анализа распределения ошибок.

## DVC Plots

DVC автоматически генерирует графики:

```bash
# Просмотр графиков
dvc plots show

# Сравнение с предыдущим запуском
dvc plots diff
```

## Интерпретация результатов

### Хорошие результаты

- R² > 0.85 — модель объясняет >85% дисперсии
- Низкий MAPE (<15%) — относительная ошибка приемлема
- Остатки ~нормально распределены (residual_mean ≈ 0)

### Признаки переобучения

- Train R² >> Val R² — большой разрыв между метриками
- Val метрики ухудшаются при увеличении сложности модели

### Текущие результаты

| Метрика | Train | Validation |
|---------|-------|------------|
| R² | 0.998 | 0.909 |
| RMSE | 3,347 | 26,405 |
| MAE | 2,699 | 15,896 |

!!! warning "Переобучение"
    Разрыв между train и validation R² (0.998 vs 0.909) указывает на некоторое переобучение. Рекомендуется:

    - Уменьшить `max_depth`
    - Увеличить `min_samples_split`
    - Использовать регуляризацию
