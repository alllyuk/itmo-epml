# Локальная установка

Руководство по локальному развертыванию проекта.

## Требования

- Python 3.10+
- Poetry 1.7+
- Git
- 4GB RAM (минимум)
- 2GB свободного места

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/alllyuk/itmo-epml
cd itmo-epml
```

### 2. Установка Poetry

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### 3. Установка зависимостей

```bash
# Все зависимости
poetry install --with dev,docs

# Только production
poetry install --only main
```

### 4. Активация окружения

```bash
poetry shell
```

### 5. Настройка pre-commit

```bash
pre-commit install
```

### 6. Загрузка данных

```bash
dvc pull
```

## Запуск

### ML Pipeline

```bash
# Полный пайплайн
dvc repro

# С мониторингом
poetry run python src/itmo_epml/main.py
```

### MLflow UI

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Доступ: [http://localhost:5000](http://localhost:5000)

### Документация

```bash
poetry run mkdocs serve
```

Доступ: [http://localhost:8000](http://localhost:8000)

### Тесты

```bash
poetry run pytest
```

## Конфигурация

### Переменные окружения

Создайте `.env` файл:

```bash title=".env"
# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# ClearML (опционально)
CLEARML_API_ACCESS_KEY=your_key
CLEARML_API_SECRET_KEY=your_secret
CLEARML_API_HOST=http://localhost:8008
```

### Параметры модели

Отредактируйте `params.yaml`:

```yaml title="params.yaml"
model:
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
```

## Структура проекта

```
itmo-epml/
├── src/itmo_epml/      # Исходный код
├── configs/            # Hydra конфигурации
├── data/
│   ├── raw/            # Исходные данные
│   ├── interim/        # Промежуточные данные
│   └── processed/      # Обработанные данные
├── models/             # Обученные модели
├── reports/            # Отчёты и метрики
├── tests/              # Тесты
├── dvc.yaml            # DVC пайплайн
├── params.yaml         # Параметры
├── pyproject.toml      # Poetry конфигурация
└── mlflow.db           # MLflow база данных
```

## Проверка установки

```bash
# Проверка Python версии
python --version  # должно быть 3.10+

# Проверка Poetry
poetry --version

# Проверка DVC
dvc version

# Проверка зависимостей
poetry show

# Запуск тестов
poetry run pytest -v

# Проверка кода
pre-commit run --all-files
```

## Типичные проблемы

### Poetry не находит Python

```bash
poetry env use python3.10
```

### DVC pull не работает

```bash
# Проверьте remote
dvc remote list

# Настройте credentials если нужно
dvc remote modify myremote access_key_id YOUR_KEY
```

### Pre-commit падает

```bash
pre-commit autoupdate
pre-commit run --all-files
```

### Не хватает памяти

Уменьшите `n_estimators` или `max_depth` в `params.yaml`.

## Обновление

```bash
# Обновить код
git pull

# Обновить зависимости
poetry update

# Обновить данные
dvc pull
```
