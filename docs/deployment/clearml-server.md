# ClearML Server

Руководство по развертыванию ClearML Server для удалённого управления экспериментами.

## Требования

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM (рекомендуется)
- 10GB свободного места

## Компоненты

ClearML Server состоит из нескольких сервисов:

| Сервис | Порт | Описание |
|--------|------|----------|
| API Server | 8008 | REST API |
| Web Server | 8080 | Веб-интерфейс |
| File Server | 8081 | Хранилище артефактов |
| MongoDB | 27017 | База данных |
| Redis | 6379 | Кэш и очереди |
| Elasticsearch | 9200 | Поиск и индексация |

## Установка

### 1. Docker Compose файл

```yaml title="docker-compose.clearml.yml"
version: "3.8"

services:
  mongodb:
    image: mongo:4.4
    volumes:
      - clearml-data-mongo:/data/db
    restart: unless-stopped

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    volumes:
      - clearml-data-elastic:/usr/share/elasticsearch/data
    restart: unless-stopped

  redis:
    image: redis:6.2
    restart: unless-stopped

  clearml-apiserver:
    image: allegroai/clearml:latest
    depends_on:
      - mongodb
      - elasticsearch
      - redis
    environment:
      - CLEARML_HOST_IP=${CLEARML_HOST_IP:-localhost}
    ports:
      - "8008:8008"
    restart: unless-stopped

  clearml-webserver:
    image: allegroai/clearml:latest
    depends_on:
      - clearml-apiserver
    ports:
      - "8080:8080"
    restart: unless-stopped

  clearml-fileserver:
    image: allegroai/clearml:latest
    depends_on:
      - clearml-apiserver
    volumes:
      - clearml-data-fileserver:/mnt/fileserver
    ports:
      - "8081:8081"
    restart: unless-stopped

volumes:
  clearml-data-mongo:
  clearml-data-elastic:
  clearml-data-fileserver:
```

### 2. Запуск сервера

```bash
docker-compose -f docker-compose.clearml.yml up -d
```

### 3. Проверка статуса

```bash
docker-compose -f docker-compose.clearml.yml ps
```

### 4. Доступ к веб-интерфейсу

Откройте [http://localhost:8080](http://localhost:8080)

## Конфигурация клиента

### 1. Создание credentials

1. Откройте [http://localhost:8080](http://localhost:8080)
2. Перейдите в **Settings** → **Workspace** → **Create new credentials**
3. Скопируйте `access_key` и `secret_key`

### 2. Настройка .env

```bash title=".env"
CLEARML_API_ACCESS_KEY=your_access_key
CLEARML_API_SECRET_KEY=your_secret_key
CLEARML_API_HOST=http://localhost:8008
CLEARML_WEB_HOST=http://localhost:8080
CLEARML_FILES_HOST=http://localhost:8081
```

### 3. Инициализация ClearML

```bash
poetry run clearml-init
```

Введите credentials при запросе.

### 4. Проверка подключения

```python
from clearml import Task

# Создание тестовой задачи
task = Task.init(
    project_name="Test",
    task_name="Connection Test"
)
print("Connected successfully!")
task.close()
```

## Использование

### Создание проекта

```python
from clearml import Task

task = Task.init(
    project_name="House Prices Prediction",
    task_name="experiment_1"
)
```

### Логирование метрик

```python
logger = task.get_logger()

# Скалярные метрики
logger.report_scalar("validation", "r2", value=0.909, iteration=1)
logger.report_scalar("validation", "mse", value=697213256.59, iteration=1)

# Графики
logger.report_scatter2d(
    "predictions",
    "actual_vs_predicted",
    scatter=[[actual, predicted] for actual, predicted in zip(y_actual, y_pred)],
    iteration=0
)
```

### Сохранение моделей

```python
from clearml import OutputModel

output_model = OutputModel(task=task, framework="scikit-learn")
output_model.update_weights("models/model.joblib")
```

### Сравнение экспериментов

```python
from itmo_epml.clearml_integration import ExperimentComparator

comparator = ExperimentComparator("House Prices Prediction")
df = comparator.compare_metrics(["val_r2", "val_mse", "val_mae"])
best = comparator.get_best_experiment("val_r2", mode="max")
```

## Управление сервером

### Остановка

```bash
docker-compose -f docker-compose.clearml.yml down
```

### Перезапуск

```bash
docker-compose -f docker-compose.clearml.yml restart
```

### Просмотр логов

```bash
docker-compose -f docker-compose.clearml.yml logs -f
```

### Очистка данных

```bash
# Остановка и удаление volumes
docker-compose -f docker-compose.clearml.yml down -v
```

## Хранение данных

Данные хранятся в Docker volumes:

| Volume | Содержимое |
|--------|------------|
| `clearml-data-mongo` | База данных MongoDB |
| `clearml-data-elastic` | Индексы Elasticsearch |
| `clearml-data-fileserver` | Файлы и артефакты |

### Backup

```bash
# Backup MongoDB
docker exec clearml-mongodb mongodump --out /backup

# Копирование backup
docker cp clearml-mongodb:/backup ./backup
```

## Интеграция с пайплайном

### Запуск с ClearML трекингом

```bash
poetry run python scripts/run_clearml_pipeline.py --mode local
```

### Grid Search

```bash
poetry run python scripts/run_clearml_pipeline.py --mode grid --experiments 15
```

### Сравнение экспериментов

```bash
poetry run python scripts/compare_experiments.py
```

## Troubleshooting

### Elasticsearch не запускается

```bash
# Увеличьте vm.max_map_count
sudo sysctl -w vm.max_map_count=262144
```

### Порты заняты

Измените порты в `docker-compose.clearml.yml`:

```yaml
ports:
  - "18080:8080"  # Вместо 8080
```

### Недостаточно памяти

Уменьшите память для Elasticsearch:

```yaml
environment:
  - ES_JAVA_OPTS=-Xms256m -Xmx256m
```
