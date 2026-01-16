# ITMO EPML - House Prices ML Pipeline

Добро пожаловать в документацию проекта курса **"Инженерные практики в Machine Learning"**.

## О проекте

Этот проект реализует production-ready ML пайплайн для предсказания цен на недвижимость с использованием современных инструментов MLOps:

- **DVC** — версионирование данных и оркестрация пайплайна
- **MLflow** — трекинг экспериментов и реестр моделей
- **ClearML** — удалённое управление экспериментами
- **Hydra** — управление конфигурациями
- **Poetry** — управление зависимостями

## Быстрые ссылки

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __Установка__

    ---

    Установите проект и все зависимости

    [:octicons-arrow-right-24: Руководство по установке](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } __Быстрый старт__

    ---

    Запустите ML пайплайн за 5 минут

    [:octicons-arrow-right-24: Быстрый старт](getting-started/quickstart.md)

-   :material-chart-line:{ .lg .middle } __Эксперименты__

    ---

    Результаты экспериментов и сравнение моделей

    [:octicons-arrow-right-24: Отчёты об экспериментах](experiments/index.md)

-   :material-cog:{ .lg .middle } __Pipeline__

    ---

    Документация ML пайплайна

    [:octicons-arrow-right-24: Обзор пайплайна](pipeline/overview.md)

</div>

## Структура проекта

```
itmo-epml/
├── src/itmo_epml/          # Исходный код
│   ├── stages/             # Стадии DVC пайплайна
│   │   ├── data_prepare.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── clearml_integration/ # Интеграция с ClearML
│   ├── main.py             # Точка входа
│   ├── model_registry.py   # Реестр моделей
│   └── monitoring.py       # Мониторинг пайплайна
├── configs/                # Hydra конфигурации
│   ├── config.yaml
│   ├── model/              # Конфиги моделей
│   ├── data/               # Конфиги данных
│   └── experiment/         # Пресеты экспериментов
├── data/                   # Данные (DVC)
│   ├── raw/                # Исходные данные
│   ├── interim/            # Подготовленные данные
│   └── processed/          # Признаки для обучения
├── models/                 # Обученные модели
├── reports/                # Отчёты и метрики
│   ├── metrics/            # JSON метрики
│   └── figures/            # Визуализации
├── tests/                  # Тесты
├── dvc.yaml                # DVC пайплайн
├── params.yaml             # Параметры DVC
└── pyproject.toml          # Poetry конфигурация
```

## Быстрый старт

```bash
# Клонирование репозитория
git clone https://github.com/alllyuk/itmo-epml
cd itmo-epml

# Установка зависимостей
poetry install

# Загрузка данных
dvc pull

# Запуск пайплайна
dvc repro

# Просмотр результатов в MLflow
poetry run mlflow ui
```

## Отчёты по домашним заданиям

| Задание | Тема | Описание |
|---------|------|----------|
| [HW1](reports/hw1.md) | Настройка проекта | Copier template, pre-commit hooks, Poetry |
| [HW2](reports/hw2.md) | Версионирование | DVC для данных и моделей |
| [HW3](reports/hw3.md) | Эксперименты | MLflow трекинг |
| [HW4](reports/hw4.md) | Автоматизация | DVC pipeline + Hydra |
| [HW5](reports/hw5.md) | ClearML | Интеграция с ClearML |

## Метрики лучшей модели

| Метрика | Значение |
|---------|----------|
| Validation R² | 0.909 |
| Validation RMSE | 26,405 |
| Validation MAE | 15,896 |
| Model | Gradient Boosting |

## Автор

**Alexey Kornelyuk**

- GitHub: [alllyuk](https://github.com/alllyuk)
