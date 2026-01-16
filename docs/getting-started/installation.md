# Установка

Руководство по установке проекта и всех необходимых зависимостей.

## Требования

- Python 3.10+
- Poetry 1.7+
- Git
- Docker (опционально, для ClearML)

## Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/alllyuk/itmo-epml
cd itmo-epml
```

## Шаг 2: Установка зависимостей

Проект использует Poetry для управления зависимостями.

```bash
# Установка всех зависимостей (включая dev и docs)
poetry install --with dev,docs

# Активация виртуального окружения
poetry shell
```

!!! tip "Poetry"
    Если Poetry не установлен, установите его:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

## Шаг 3: Установка pre-commit hooks

```bash
pre-commit install
```

Это установит следующие hooks:
- `black` — форматирование кода
- `ruff` — линтинг
- `mypy` — проверка типов
- `bandit` — проверка безопасности

## Шаг 4: Загрузка данных через DVC

```bash
dvc pull
```

!!! note "DVC Remote"
    Данные хранятся в удалённом хранилище. Убедитесь, что у вас есть доступ к DVC remote.

## Шаг 5: Проверка установки

```bash
# Запуск тестов
poetry run pytest

# Проверка качества кода
pre-commit run --all-files

# Проверка DVC пайплайна
dvc status
```

## Опционально: Настройка ClearML

Для работы с ClearML необходимо запустить сервер:

```bash
# Запуск ClearML Server
docker-compose -f docker-compose.clearml.yml up -d
```

Затем инициализируйте ClearML:

```bash
poetry run clearml-init
```

Подробнее см. [Развертывание ClearML](../deployment/clearml-server.md).

## Структура зависимостей

```toml
[tool.poetry.dependencies]
python = "^3.10"
numpy = "^1.26.0"
pandas = "^2.1.0"
scikit-learn = "^1.3.0"
matplotlib = "^3.8.0"
seaborn = "^0.13.0"
hydra-core = "^1.3.0"
dvc = "^3.64.2"
mlflow = "^3.7.0"
clearml = "^2.1.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pre-commit = "^3.6.0"
black = "^23.12.0"
ruff = "^0.1.9"
mypy = "^1.8.0"

[tool.poetry.group.docs.dependencies]
mkdocs = "^1.6.1"
mkdocs-material = "^9.7.1"
```

## Возможные проблемы

### Poetry не находит Python 3.10

```bash
poetry env use python3.10
```

### DVC pull не работает

Убедитесь, что настроен доступ к remote:

```bash
dvc remote list
```

### Pre-commit hooks падают

Попробуйте обновить hooks:

```bash
pre-commit autoupdate
pre-commit run --all-files
```
