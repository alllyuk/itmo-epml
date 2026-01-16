# Feature Engineering

Стадия `feature_engineering` создаёт признаки для обучения модели.

## Описание

Эта стадия выполняет:

1. Разделение на train/validation наборы
2. Обработку пропущенных значений
3. Кодирование категориальных признаков
4. Масштабирование числовых признаков
5. Сохранение трансформеров

## Входные данные

| Файл | Описание |
|------|----------|
| `data/interim/train_prepared.csv` | Очищенные тренировочные данные |
| `data/interim/test_prepared.csv` | Очищенные тестовые данные |

## Выходные данные

| Файл | Описание |
|------|----------|
| `data/processed/X_train.csv` | Признаки для обучения |
| `data/processed/X_val.csv` | Признаки для валидации |
| `data/processed/y_train.csv` | Целевая переменная (train) |
| `data/processed/y_val.csv` | Целевая переменная (val) |
| `data/processed/X_test.csv` | Признаки для теста |
| `models/scaler.joblib` | Обученный scaler |
| `models/feature_columns.json` | Список признаков |

## Реализация

```python title="src/itmo_epml/stages/feature_engineering.py"
@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Загрузка данных
    train_df = pd.read_csv(cfg.data.interim_train_path)
    test_df = pd.read_csv(cfg.data.interim_test_path)

    # Разделение на X и y
    X = train_df.drop(columns=[cfg.data.target_col])
    y = train_df[cfg.data.target_col]

    # Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state
    )

    # Обработка признаков
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Заполнение пропусков
    X_train[numeric_features] = X_train[numeric_features].fillna(
        X_train[numeric_features].median()
    )
    X_train[categorical_features] = X_train[categorical_features].fillna('Missing')

    # One-hot encoding
    X_train = pd.get_dummies(X_train, columns=categorical_features)
    X_val = pd.get_dummies(X_val, columns=categorical_features)

    # Выравнивание колонок
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Сохранение
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        "data/processed/X_train.csv", index=False
    )
    joblib.dump(scaler, "models/scaler.joblib")
```

## Конфигурация

```yaml title="params.yaml"
data:
  target_col: SalePrice
  test_size: 0.2
  random_state: 42
```

## Запуск

```bash
# Через DVC
dvc repro feature_engineering

# Напрямую
python -m src.itmo_epml.stages.feature_engineering
```

## Статистика признаков

После обработки:

| Метрика | Значение |
|---------|----------|
| Тренировочных записей | 1168 |
| Валидационных записей | 292 |
| Количество признаков | 242 |

## Обработка признаков

### Числовые признаки

- Заполнение пропусков медианой
- Стандартизация (StandardScaler)

### Категориальные признаки

- Заполнение пропусков значением "Missing"
- One-hot encoding

### Важные признаки

Топ-10 признаков по важности (после обучения модели):

1. `OverallQual` — общее качество
2. `GrLivArea` — жилая площадь
3. `TotalBsmtSF` — площадь подвала
4. `GarageCars` — вместимость гаража
5. `YearBuilt` — год постройки
6. `FullBath` — количество ванных
7. `GarageArea` — площадь гаража
8. `1stFlrSF` — площадь первого этажа
9. `TotRmsAbvGrd` — всего комнат
10. `YearRemodAdd` — год ремонта
