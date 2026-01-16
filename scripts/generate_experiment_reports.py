#!/usr/bin/env python3
"""
Генерация отчётов об экспериментах с визуализациями для MkDocs документации.

Этот скрипт читает данные экспериментов из reports/metrics/ и генерирует:
1. Markdown отчёты с графиками
2. Сравнительные таблицы
3. Отчёт о лучших моделях
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# Для визуализаций
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend для CI
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn не установлены, графики не будут сгенерированы")


class ExperimentReportGenerator:
    """Генератор документации экспериментов из данных метрик."""

    def __init__(
        self,
        metrics_dir: str = "reports/metrics",
        output_dir: str = "docs/experiments",
        assets_dir: str = "docs/assets/images/experiments",
    ):
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path(output_dir)
        self.assets_dir = Path(assets_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def load_experiments_summary(self) -> list:
        """Загрузка сводки экспериментов из JSON."""
        summary_path = self.metrics_dir / "experiments_summary.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def load_train_metrics(self) -> dict:
        """Загрузка метрик обучения."""
        path = self.metrics_dir / "train_metrics.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def load_eval_metrics(self) -> dict:
        """Загрузка метрик оценки."""
        path = self.metrics_dir / "eval_metrics.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def load_data_stats(self) -> dict:
        """Загрузка статистики данных."""
        path = self.metrics_dir / "data_stats.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def generate_comparison_chart(self, experiments: list) -> str:
        """Генерация графика сравнения экспериментов."""
        if not PLOTTING_AVAILABLE or not experiments:
            return ""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Сравнение экспериментов", fontsize=14, fontweight="bold")

        # Подготовка данных
        metrics_df = pd.DataFrame([e["metrics"] for e in experiments])
        metrics_df["experiment"] = [e["experiment"] for e in experiments]
        metrics_df = metrics_df.sort_values("val_r2", ascending=True)

        # 1. R2 Score comparison
        ax1 = axes[0, 0]
        colors = plt.cm.RdYlGn(
            (metrics_df["val_r2"] - metrics_df["val_r2"].min())
            / (metrics_df["val_r2"].max() - metrics_df["val_r2"].min() + 1e-10)
        )
        ax1.barh(range(len(metrics_df)), metrics_df["val_r2"], color=colors)
        ax1.set_yticks(range(len(metrics_df)))
        ax1.set_yticklabels(
            [e.replace("experiment_", "E") for e in metrics_df["experiment"]],
            fontsize=8,
        )
        ax1.set_xlabel("Validation R² Score")
        ax1.set_title("R² Score по экспериментам")
        ax1.axvline(x=0.9, color="red", linestyle="--", alpha=0.7, label="Цель: 0.9")
        ax1.legend()

        # 2. MSE comparison
        ax2 = axes[0, 1]
        ax2.barh(
            range(len(metrics_df)), metrics_df["val_mse"] / 1e6, color="steelblue"
        )
        ax2.set_yticks(range(len(metrics_df)))
        ax2.set_yticklabels(
            [e.replace("experiment_", "E") for e in metrics_df["experiment"]],
            fontsize=8,
        )
        ax2.set_xlabel("Validation MSE (миллионы)")
        ax2.set_title("MSE по экспериментам")

        # 3. Hyperparameter impact on R2
        ax3 = axes[1, 0]
        params_df = pd.DataFrame([e["params"] for e in experiments])
        params_df["val_r2"] = [e["metrics"]["val_r2"] for e in experiments]

        for depth in sorted(params_df["max_depth"].unique()):
            subset = params_df[params_df["max_depth"] == depth]
            ax3.plot(
                subset["n_estimators"],
                subset["val_r2"],
                "o-",
                label=f"depth={depth}",
                markersize=8,
            )
        ax3.set_xlabel("Количество деревьев (n_estimators)")
        ax3.set_ylabel("Validation R²")
        ax3.set_title("Влияние гиперпараметров на R²")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Train vs Validation R2
        ax4 = axes[1, 1]
        train_r2 = [e["metrics"]["train_r2"] for e in experiments]
        val_r2 = [e["metrics"]["val_r2"] for e in experiments]
        ax4.scatter(train_r2, val_r2, c="steelblue", alpha=0.7, s=100)
        ax4.plot([0.85, 1.0], [0.85, 1.0], "r--", alpha=0.5, label="Идеальная линия")
        ax4.set_xlabel("Train R²")
        ax4.set_ylabel("Validation R²")
        ax4.set_title("Train vs Validation R² (проверка переобучения)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        chart_path = self.assets_dir / "experiment_comparison.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        return "../assets/images/experiments/experiment_comparison.png"

    def generate_metrics_heatmap(self, experiments: list) -> str:
        """Генерация heatmap метрик."""
        if not PLOTTING_AVAILABLE or not experiments:
            return ""

        metrics_data = []
        for exp in experiments:
            row = {
                "Эксперимент": exp["experiment"].replace("experiment_", "E"),
                "Train R²": exp["metrics"]["train_r2"],
                "Val R²": exp["metrics"]["val_r2"],
                "Train MAE (тыс.)": exp["metrics"]["train_mae"] / 1000,
                "Val MAE (тыс.)": exp["metrics"]["val_mae"] / 1000,
            }
            metrics_data.append(row)

        df = pd.DataFrame(metrics_data).set_index("Эксперимент")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", center=0.9, ax=ax)
        ax.set_title("Heatmap метрик экспериментов\n(MAE в тысячах)")

        chart_path = self.assets_dir / "metrics_heatmap.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        return "../assets/images/experiments/metrics_heatmap.png"

    def generate_comparison_report(self) -> str:
        """Генерация отчёта comparison.md."""
        experiments = self.load_experiments_summary()

        content = [
            "# Сравнение экспериментов",
            "",
            f"*Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Обзор",
            "",
            f"Этот отчёт сравнивает **{len(experiments)}** экспериментов "
            "для модели предсказания цен на недвижимость.",
            "",
        ]

        if experiments:
            # Генерация графиков
            comparison_chart = self.generate_comparison_chart(experiments)
            heatmap_chart = self.generate_metrics_heatmap(experiments)

            if comparison_chart:
                content.extend(
                    [
                        "## Визуальное сравнение",
                        "",
                        f"![Сравнение экспериментов]({comparison_chart})",
                        "",
                    ]
                )

            if heatmap_chart:
                content.extend(
                    [
                        "## Heatmap метрик",
                        "",
                        f"![Heatmap метрик]({heatmap_chart})",
                        "",
                    ]
                )

            # Таблица экспериментов
            content.extend(
                [
                    "## Сводная таблица экспериментов",
                    "",
                    "| Эксперимент | n_estimators | max_depth | learning_rate | Val R² | Val MSE | Val MAE |",
                    "|-------------|--------------|-----------|---------------|--------|---------|---------|",
                ]
            )

            sorted_experiments = sorted(
                experiments, key=lambda x: x["metrics"]["val_r2"], reverse=True
            )

            for exp in sorted_experiments:
                params = exp["params"]
                metrics = exp["metrics"]
                content.append(
                    f"| {exp['experiment']} | {params.get('n_estimators', 'N/A')} | "
                    f"{params.get('max_depth', 'N/A')} | {params.get('learning_rate', 'N/A')} | "
                    f"{metrics['val_r2']:.4f} | {metrics['val_mse']:,.0f} | {metrics['val_mae']:,.2f} |"
                )

            # Лучший эксперимент
            best = sorted_experiments[0]
            content.extend(
                [
                    "",
                    "## Лучший эксперимент",
                    "",
                    f"**{best['experiment']}** показал лучший результат по Validation R².",
                    "",
                    "### Параметры",
                    "",
                    "```yaml",
                ]
            )
            for key, value in best["params"].items():
                content.append(f"{key}: {value}")
            content.extend(
                [
                    "```",
                    "",
                    "### Метрики",
                    "",
                    "| Метрика | Значение |",
                    "|---------|----------|",
                ]
            )
            for key, value in best["metrics"].items():
                if isinstance(value, float):
                    content.append(f"| {key} | {value:,.4f} |")
                else:
                    content.append(f"| {key} | {value} |")

            # Ключевые выводы
            content.extend(
                [
                    "",
                    "## Ключевые выводы",
                    "",
                    "1. **Лучшая конфигурация**: "
                    f"n_estimators={best['params'].get('n_estimators')}, "
                    f"max_depth={best['params'].get('max_depth')}, "
                    f"learning_rate={best['params'].get('learning_rate')}",
                    "",
                    f"2. **Лучший Validation R²**: {best['metrics']['val_r2']:.4f}",
                    "",
                    f"3. **Диапазон R²**: {min(e['metrics']['val_r2'] for e in experiments):.4f} - "
                    f"{max(e['metrics']['val_r2'] for e in experiments):.4f}",
                    "",
                    "4. **Наблюдения**:",
                    "    - Увеличение `max_depth` улучшает train R², но может вести к переобучению",
                    "    - Learning rate 0.1 работает лучше, чем 0.05",
                    "    - Больше деревьев (200) с умеренной глубиной (5) даёт лучшую генерализацию",
                ]
            )

        report_path = self.output_dir / "comparison.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

        return str(report_path)

    def generate_best_models_report(self) -> str:
        """Генерация отчёта best-models.md."""
        experiments = self.load_experiments_summary()
        train_metrics = self.load_train_metrics()
        eval_metrics = self.load_eval_metrics()

        content = [
            "# Лучшие модели",
            "",
            f"*Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Текущая production модель",
            "",
        ]

        if train_metrics or eval_metrics:
            content.extend(
                [
                    "### Метрики обучения",
                    "",
                    "| Метрика | Значение |",
                    "|---------|----------|",
                ]
            )
            for key, value in train_metrics.items():
                if key != "run_id":
                    if isinstance(value, float):
                        content.append(f"| {key} | {value:,.4f} |")
                    else:
                        content.append(f"| {key} | {value} |")

            content.extend(
                [
                    "",
                    "### Метрики оценки",
                    "",
                    "| Метрика | Значение |",
                    "|---------|----------|",
                ]
            )
            for key, value in eval_metrics.items():
                if isinstance(value, float):
                    content.append(f"| {key} | {value:,.4f} |")
                else:
                    content.append(f"| {key} | {value} |")

        # Топ-5 экспериментов
        if experiments:
            sorted_experiments = sorted(
                experiments, key=lambda x: x["metrics"]["val_r2"], reverse=True
            )[:5]

            content.extend(
                [
                    "",
                    "## Топ-5 экспериментов по Validation R²",
                    "",
                ]
            )

            for i, exp in enumerate(sorted_experiments, 1):
                content.extend(
                    [
                        f"### #{i}: {exp['experiment']}",
                        "",
                        f"- **Validation R²**: {exp['metrics']['val_r2']:.4f}",
                        f"- **Validation MSE**: {exp['metrics']['val_mse']:,.2f}",
                        f"- **Validation MAE**: {exp['metrics']['val_mae']:,.2f}",
                        f"- **Параметры**: n_estimators={exp['params'].get('n_estimators')}, "
                        f"max_depth={exp['params'].get('max_depth')}, "
                        f"learning_rate={exp['params'].get('learning_rate')}",
                        "",
                    ]
                )

        content.extend(
            [
                "## Критерии выбора модели",
                "",
                "Модели ранжируются по **Validation R²** с учётом:",
                "",
                "- **R² Score**: Основная метрика, измеряющая объяснённую дисперсию",
                "- **MSE**: Mean Squared Error для чувствительности к большим ошибкам",
                "- **MAE**: Mean Absolute Error для интерпретируемости",
                "- **Переобучение**: Разрыв между train и validation метриками",
                "",
                "## Воспроизведение лучшей модели",
                "",
                "```bash",
                "# Через DVC пайплайн с оптимизированными параметрами",
                "dvc repro",
                "",
                "# Или с конкретными параметрами через Hydra",
                "python -m src.itmo_epml.run_pipeline model.n_estimators=200 model.max_depth=5",
                "```",
            ]
        )

        report_path = self.output_dir / "best-models.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

        return str(report_path)

    def generate_all_reports(self):
        """Генерация всех отчётов об экспериментах."""
        print("Генерация отчётов об экспериментах...")

        reports = [
            ("Сравнение", self.generate_comparison_report()),
            ("Лучшие модели", self.generate_best_models_report()),
        ]

        for name, path in reports:
            print(f"  Сгенерировано: {name} -> {path}")

        print("Готово!")


def main():
    """Точка входа."""
    generator = ExperimentReportGenerator()
    generator.generate_all_reports()


if __name__ == "__main__":
    main()
