# Обзор экспериментов

Этот раздел документирует все ML эксперименты проекта House Prices Prediction.

## Статистика

| Показатель | Значение |
|------------|----------|
| Всего экспериментов | 15+ |
| Лучший Validation R² | 0.909 |
| Тип модели | Gradient Boosting Regressor |
| Framework | scikit-learn |

## Датасет

| Характеристика | Значение |
|----------------|----------|
| Тренировочных записей | 1460 |
| Тестовых записей | 1459 |
| Признаков (raw) | 81 |
| Признаков (после обработки) | 242 |
| Целевая переменная | SalePrice |

## Инструменты трекинга

Проект использует несколько систем трекинга экспериментов:

### MLflow

- **Локальный трекинг** экспериментов
- **Model Registry** для версионирования моделей
- **Автологирование** параметров и метрик

### ClearML

- **Удалённое управление** экспериментами
- **Pipeline orchestration**
- **Сравнение экспериментов** через веб-интерфейс

### DVC

- **Версионирование данных** и моделей
- **Воспроизводимость** экспериментов
- **Метрики** как часть пайплайна

## Отчёты

- [Сравнение экспериментов](comparison.md) — детальное сравнение всех экспериментов
- [Лучшие модели](best-models.md) — топ моделей и критерии выбора
- [Трекинг экспериментов](tracking.md) — настройка MLflow и ClearML

## Запуск новых экспериментов

### Одиночный эксперимент

```bash
# Базовый запуск
python -m src.itmo_epml.run_pipeline

# С изменением параметров
python -m src.itmo_epml.run_pipeline \
    model.n_estimators=200 \
    model.max_depth=10
```

### Grid Search (Hydra)

```bash
python -m src.itmo_epml.stages.train --multirun \
    model.n_estimators=50,100,200 \
    model.max_depth=5,10,20
```

### ClearML Pipeline

```bash
# Локальный запуск с трекингом
poetry run python scripts/run_clearml_pipeline.py --mode local

# Grid search через ClearML
poetry run python scripts/run_clearml_pipeline.py --mode grid --experiments 15
```

## Метрики экспериментов

Для каждого эксперимента логируются:

| Метрика | Описание |
|---------|----------|
| `train_mse` | Mean Squared Error на обучении |
| `train_rmse` | Root MSE на обучении |
| `train_mae` | Mean Absolute Error на обучении |
| `train_r2` | R² Score на обучении |
| `val_mse` | Mean Squared Error на валидации |
| `val_rmse` | Root MSE на валидации |
| `val_mae` | Mean Absolute Error на валидации |
| `val_r2` | R² Score на валидации |

## Гиперпараметры

Исследуемые гиперпараметры:

| Параметр | Значения | Лучшее |
|----------|----------|--------|
| `n_estimators` | 50, 100, 200 | 200 |
| `max_depth` | 5, 10, 15, 20 | 5 |
| `learning_rate` | 0.05, 0.1 | 0.1 |
| `min_samples_split` | 2, 5 | 2 |

## Ключевые выводы

1. **Лучшая конфигурация**: `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`

2. **Влияние `max_depth`**: Увеличение глубины улучшает train R², но ведёт к переобучению

3. **Влияние `learning_rate`**: 0.1 даёт лучшие результаты, чем 0.05

4. **Влияние `n_estimators`**: Больше деревьев (200) с умеренной глубиной (5) даёт лучшую генерализацию
