# Docker

Руководство по запуску проекта в Docker контейнере.

## Требования

- Docker 20.10+
- Docker Compose 2.0+ (для ClearML)
- 4GB RAM

## Dockerfile

```dockerfile title="Dockerfile"
# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Установка Poetry
RUN pip install poetry==1.7.1

# Копирование файлов зависимостей
COPY pyproject.toml poetry.lock ./

# Установка зависимостей
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Копирование установленных пакетов
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копирование кода проекта
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/
COPY params.yaml dvc.yaml ./

# Команда по умолчанию
CMD ["python", "-m", "src.itmo_epml.main"]
```

## Сборка образа

```bash
docker build -t itmo-epml .
```

## Запуск контейнера

### Базовый запуск

```bash
docker run -it itmo-epml
```

### С монтированием данных

```bash
docker run -it \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/reports:/app/reports \
    itmo-epml
```

### С переменными окружения

```bash
docker run -it \
    -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    -v $(pwd)/data:/app/data \
    itmo-epml
```

### Интерактивный режим

```bash
docker run -it itmo-epml bash
```

## Docker Compose

```yaml title="docker-compose.yml"
version: '3.8'

services:
  ml-pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
      - ./mlflow.db:/app/mlflow.db
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    command: python -m src.itmo_epml.main

  mlflow:
    image: python:3.10-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow.db:/app/mlflow.db
      - ./mlruns:/app/mlruns
    working_dir: /app
    command: >
      bash -c "pip install mlflow &&
               mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db"
```

### Запуск с Docker Compose

```bash
# Запуск всех сервисов
docker-compose up -d

# Только ML пайплайн
docker-compose up ml-pipeline

# Только MLflow UI
docker-compose up mlflow
```

## Multi-stage build

Для оптимизации размера образа используется multi-stage build:

1. **Builder stage**: Устанавливает зависимости
2. **Runtime stage**: Копирует только необходимое

Это уменьшает размер образа примерно на 50%.

## Полезные команды

```bash
# Просмотр логов
docker logs <container_id>

# Вход в контейнер
docker exec -it <container_id> bash

# Остановка
docker stop <container_id>

# Удаление
docker rm <container_id>

# Очистка
docker system prune
```

## Оптимизация

### .dockerignore

```text title=".dockerignore"
.git
.venv
venv
__pycache__
*.pyc
.env
mlruns/
outputs/
multirun/
.pytest_cache/
.mypy_cache/
```

### Кэширование слоёв

Poetry файлы копируются отдельно для кэширования зависимостей:

```dockerfile
COPY pyproject.toml poetry.lock ./
RUN poetry install

COPY . .  # Код копируется после установки зависимостей
```

## CI/CD интеграция

```yaml title=".github/workflows/docker.yml"
name: Docker Build

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t itmo-epml .

      - name: Run tests in Docker
        run: docker run itmo-epml pytest
```
