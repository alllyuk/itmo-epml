# ITMO EPML - House Prices ML Pipeline

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://alllyuk.github.io/itmo-epml/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.7+-blue.svg)](https://python-poetry.org/)

Репозиторий курса "Engineering practices in Machine Learning" (Инженерные практики в ML).

**[Полная документация](https://alllyuk.github.io/itmo-epml/)**

## Метрики лучшей модели

| Метрика | Значение |
|---------|----------|
| Validation R² | 0.909 |
| Validation RMSE | 26,405 |
| Validation MAE | 15,896 |
| Model | Gradient Boosting |

## Документация

- [HW Report 1](./reports/REPORT1.md) - Настройка проекта и структура
- [HW Report 2](./reports/REPORT2.md) - Версионирование данных и моделей (DVC)
- [HW Report 3](./reports/REPORT3.md) - Трекинг экспериментов (MLflow)
- [HW Report 4](./reports/REPORT4.md) - Автоматизация ML пайплайна (DVC + Hydra)
- [HW Report 5](./reports/REPORT5.md) - Интеграция с ClearML

## Быстрый старт

### Требования

- Python 3.10+
- Poetry 1.7+
- Git
- DVC 3.x
- Docker (опционально)

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/alllyuk/itmo-epml
cd itmo-epml

# Установка зависимостей
poetry install

# Активация окружения
poetry shell

# Установка pre-commit hooks
pre-commit install

# Загрузка данных через DVC
dvc pull
```

### Запуск ML пайплайна

```bash
# Полный пайплайн через DVC
dvc repro

# Или с мониторингом
poetry run python src/itmo_epml/main.py

# Просмотр MLflow dashboard
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Воспроизведение результатов

### Шаг 1: Подготовка окружения

```bash
# Клонирование и установка
git clone https://github.com/alllyuk/itmo-epml
cd itmo-epml
poetry install
poetry shell

# Загрузка данных
dvc pull
```

### Шаг 2: Запуск пайплайна

```bash
# Воспроизведение полного пайплайна
dvc repro

# Просмотр метрик
dvc metrics show
```

### Шаг 3: Проверка результатов

```bash
# Тесты
poetry run pytest

# Метрики
cat reports/metrics/eval_metrics.json

# MLflow UI
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Воспроизведение конкретного эксперимента

```bash
# С определёнными параметрами
python -m src.itmo_epml.run_pipeline \
    model.n_estimators=200 \
    model.max_depth=5 \
    model.learning_rate=0.1

# Grid Search
python -m src.itmo_epml.stages.train --multirun \
    model.n_estimators=50,100,200 \
    model.max_depth=5,10,20
```

## Локальная документация

```bash
# Генерация отчётов об экспериментах
poetry run python scripts/generate_experiment_reports.py

# Запуск сервера документации
poetry run mkdocs serve

# Сборка статического сайта
poetry run mkdocs build
```

## Hydra конфигурация

```bash
# Запуск с default конфигом
python -m src.itmo_epml.run_pipeline

# Выбор модели
python -m src.itmo_epml.run_pipeline model=random_forest

# Выбор эксперимента
python -m src.itmo_epml.run_pipeline experiment=optimized

# Изменение параметров
python -m src.itmo_epml.run_pipeline model.n_estimators=200 model.max_depth=10
```

## ClearML интеграция

```bash
# Запуск ClearML Server
docker-compose -f docker-compose.clearml.yml up -d

# Запуск пайплайна с ClearML
python scripts/run_clearml_pipeline.py --mode local

# Grid search
python scripts/run_clearml_pipeline.py --mode grid --experiments 15

# Сравнение экспериментов
python scripts/compare_experiments.py

# Веб-интерфейс
open http://localhost:8080
```

## Тестирование

```bash
poetry run pytest
```

## Проверка качества кода

```bash
# Все pre-commit hooks
pre-commit run --all-files

# Форматирование
poetry run black src/ tests/

# Линтинг
poetry run ruff check src/ tests/

# Проверка безопасности
poetry run bandit -r src/ -c pyproject.toml
```

## Структура проекта

```
itmo-epml/
├── src/itmo_epml/          # Исходный код
│   ├── stages/             # Стадии DVC пайплайна
│   ├── clearml_integration/ # ClearML интеграция
│   ├── main.py             # Точка входа
│   ├── model_registry.py   # Реестр моделей
│   └── monitoring.py       # Мониторинг
├── configs/                # Hydra конфигурации
├── data/
│   ├── raw/                # Исходные данные (DVC)
│   ├── interim/            # Промежуточные данные
│   └── processed/          # Обработанные данные
├── models/                 # Обученные модели
├── reports/
│   ├── metrics/            # JSON метрики
│   └── figures/            # Визуализации
├── docs/                   # MkDocs документация
├── tests/                  # Тесты
├── scripts/                # Утилиты
├── dvc.yaml                # DVC пайплайн
├── params.yaml             # Параметры
├── mkdocs.yml              # Конфигурация документации
└── pyproject.toml          # Poetry конфигурация
```

## Docker

```bash
# Сборка образа
docker build -t itmo-epml .

# Запуск контейнера
docker run -it itmo-epml
```

## Branching strategy

| Ветка | Назначение |
|-------|------------|
| main | Стабильная версия |
| develop | Интеграция фич |
| feature/* | Новые функции |
| hotfix/* | Срочные исправления |
| hwN | Домашнее задание N |

## Автор

Alexey Kornelyuk
