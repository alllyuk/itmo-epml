# Лучшие модели

*Сгенерировано: 2026-01-16 23:28:40*

## Текущая production модель

### Метрики обучения

| Метрика | Значение |
|---------|----------|
| train_mse | 41,449,485.5071 |
| train_rmse | 6,438.1275 |
| train_mae | 5,074.1127 |
| train_r2 | 0.9931 |
| val_mse | 712,662,125.5828 |
| val_rmse | 26,695.7323 |
| val_mae | 16,060.1683 |
| val_r2 | 0.9071 |

### Метрики оценки

| Метрика | Значение |
|---------|----------|
| eval_mse | 712,662,125.5828 |
| eval_rmse | 26,695.7323 |
| eval_mae | 16,060.1683 |
| eval_r2 | 0.9071 |
| eval_mape | 9.7511 |
| residual_mean | 511.3590 |
| residual_std | 26,690.8343 |

## Топ-5 экспериментов по Validation R²

### #1: experiment_14_est200_depth5

- **Validation R²**: 0.9091
- **Validation MSE**: 697,213,256.59
- **Validation MAE**: 15,895.85
- **Параметры**: n_estimators=200, max_depth=5, learning_rate=0.1

### #2: experiment_8_est100_depth5

- **Validation R²**: 0.9071
- **Validation MSE**: 712,662,125.58
- **Validation MAE**: 16,060.17
- **Параметры**: n_estimators=100, max_depth=5, learning_rate=0.1

### #3: experiment_13_est200_depth5

- **Validation R²**: 0.9060
- **Validation MSE**: 720,881,646.41
- **Validation MAE**: 16,331.94
- **Параметры**: n_estimators=200, max_depth=5, learning_rate=0.05

### #4: experiment_2_est50_depth5

- **Validation R²**: 0.9039
- **Validation MSE**: 737,079,374.84
- **Validation MAE**: 16,656.05
- **Параметры**: n_estimators=50, max_depth=5, learning_rate=0.1

### #5: experiment_7_est100_depth5

- **Validation R²**: 0.9024
- **Validation MSE**: 749,003,912.99
- **Validation MAE**: 16,808.77
- **Параметры**: n_estimators=100, max_depth=5, learning_rate=0.05

## Критерии выбора модели

Модели ранжируются по **Validation R²** с учётом:

- **R² Score**: Основная метрика, измеряющая объяснённую дисперсию
- **MSE**: Mean Squared Error для чувствительности к большим ошибкам
- **MAE**: Mean Absolute Error для интерпретируемости
- **Переобучение**: Разрыв между train и validation метриками

## Воспроизведение лучшей модели

```bash
# Через DVC пайплайн с оптимизированными параметрами
dvc repro

# Или с конкретными параметрами через Hydra
python -m src.itmo_epml.run_pipeline model.n_estimators=200 model.max_depth=5
```