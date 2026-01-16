# Подготовка данных

Стадия `data_prepare` загружает и очищает исходные данные.

## Описание

Эта стадия выполняет:

1. Загрузку данных из CSV файлов
2. Анализ пропущенных значений
3. Удаление дубликатов
4. Базовую очистку данных
5. Сохранение статистики

## Входные данные

| Файл | Описание |
|------|----------|
| `data/raw/train.csv` | Тренировочные данные (1460 записей) |
| `data/raw/test.csv` | Тестовые данные (1459 записей) |

## Выходные данные

| Файл | Описание |
|------|----------|
| `data/interim/train_prepared.csv` | Очищенные тренировочные данные |
| `data/interim/test_prepared.csv` | Очищенные тестовые данные |
| `reports/metrics/data_stats.json` | Статистика данных |

## Реализация

```python title="src/itmo_epml/stages/data_prepare.py"
@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Загрузка данных
    train_df = pd.read_csv(cfg.data.raw_train_path)
    test_df = pd.read_csv(cfg.data.raw_test_path)

    # Анализ пропусков
    missing_train = train_df.isnull().sum()
    missing_test = test_df.isnull().sum()

    # Удаление дубликатов
    train_df = train_df.drop_duplicates()

    # Статистика
    stats = {
        "train_rows_original": len(train_df),
        "test_rows": len(test_df),
        "train_cols": len(train_df.columns),
        "columns_with_missing": int((missing_train > 0).sum()),
        "train_missing_total": int(missing_train.sum()),
    }

    # Сохранение
    train_df.to_csv(cfg.data.interim_train_path, index=False)
    test_df.to_csv(cfg.data.interim_test_path, index=False)

    with open("reports/metrics/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
```

## Статистика данных

Пример выходного файла `data_stats.json`:

```json
{
  "train_rows_original": 1460,
  "train_rows_after_dedup": 1460,
  "train_duplicates": 0,
  "test_rows": 1459,
  "test_rows_after_dedup": 1459,
  "train_cols": 81,
  "columns_with_missing": 19,
  "train_missing_total": 7829
}
```

## Конфигурация

```yaml title="configs/data/default.yaml"
raw_train_path: "data/raw/train.csv"
raw_test_path: "data/raw/test.csv"
interim_train_path: "data/interim/train_prepared.csv"
interim_test_path: "data/interim/test_prepared.csv"
```

## Запуск

```bash
# Через DVC
dvc repro data_prepare

# Напрямую
python -m src.itmo_epml.stages.data_prepare
```

## Описание датасета

**House Prices Dataset** (Kaggle) содержит информацию о домах в Эймсе, штат Айова:

- **81 признак** описывающий характеристики домов
- **1460 тренировочных** записей
- **1459 тестовых** записей
- **Целевая переменная**: `SalePrice` — цена продажи

### Типы признаков

| Тип | Количество | Примеры |
|-----|------------|---------|
| Числовые | 38 | `LotArea`, `YearBuilt`, `GrLivArea` |
| Категориальные | 43 | `MSZoning`, `Neighborhood`, `HouseStyle` |

### Признаки с пропусками

Наибольшее количество пропусков:

- `PoolQC` — 99.5%
- `MiscFeature` — 96.3%
- `Alley` — 93.8%
- `Fence` — 80.8%
- `FireplaceQu` — 47.3%
