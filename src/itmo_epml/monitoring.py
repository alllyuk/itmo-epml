"""Pipeline monitoring utilities."""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    status: str = "running"
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None


class PipelineMonitor:
    """Monitor for tracking pipeline execution."""

    def __init__(self, pipeline_name: str = "ml_pipeline"):
        self.pipeline_name = pipeline_name
        self.stages: dict[str, StageMetrics] = {}
        self.start_time = datetime.now()
        self.reports_dir = Path("reports/monitoring")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def start_stage(self, stage_name: str) -> None:
        """Mark stage as started."""
        self.stages[stage_name] = StageMetrics(name=stage_name)
        logger.info(f"[MONITOR] Stage '{stage_name}' started")

    def end_stage(
        self,
        stage_name: str,
        status: str = "success",
        metrics: Optional[dict] = None,
        error: Optional[str] = None
    ) -> None:
        """Mark stage as completed."""
        if stage_name not in self.stages:
            logger.warning(f"Stage '{stage_name}' was not started")
            return

        stage = self.stages[stage_name]
        stage.end_time = datetime.now()
        stage.duration_seconds = (stage.end_time - stage.start_time).total_seconds()
        stage.status = status
        stage.metrics = metrics or {}
        stage.error = error

        logger.info(
            f"[MONITOR] Stage '{stage_name}' completed: "
            f"status={status}, duration={stage.duration_seconds:.2f}s"
        )

    def stage_decorator(self, stage_name: str) -> Callable:
        """Decorator for automatic stage monitoring."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs) -> Any:
                self.start_stage(stage_name)
                try:
                    result = func(*args, **kwargs)
                    metrics = result if isinstance(result, dict) else {}
                    self.end_stage(stage_name, status="success", metrics=metrics)
                    return result
                except Exception as e:
                    self.end_stage(stage_name, status="failed", error=str(e))
                    raise
            return wrapper
        return decorator

    def get_summary(self) -> dict:
        """Get pipeline execution summary."""
        total_duration = (datetime.now() - self.start_time).total_seconds()

        stages_summary = []
        for name, stage in self.stages.items():
            stages_summary.append({
                "name": name,
                "status": stage.status,
                "duration_seconds": stage.duration_seconds,
                "metrics": stage.metrics,
                "error": stage.error,
            })

        failed_stages = [s for s in stages_summary if s["status"] == "failed"]

        return {
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time.isoformat(),
            "total_duration_seconds": total_duration,
            "stages_count": len(self.stages),
            "failed_count": len(failed_stages),
            "overall_status": "failed" if failed_stages else "success",
            "stages": stages_summary,
        }

    def save_report(self, filename: Optional[str] = None) -> str:
        """Save monitoring report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_report_{timestamp}.json"

        report_path = self.reports_dir / filename
        summary = self.get_summary()

        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"[MONITOR] Report saved to {report_path}")
        return str(report_path)

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print(f"Pipeline: {summary['pipeline_name']}")
        print(f"Status: {summary['overall_status'].upper()}")
        print(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
        print("=" * 60)

        for stage in summary["stages"]:
            status_icon = "✓" if stage["status"] == "success" else "✗"
            print(f"  {status_icon} {stage['name']}: {stage['duration_seconds']:.2f}s")
            if stage["error"]:
                print(f"    Error: {stage['error']}")

        print("=" * 60 + "\n")


# Global monitor instance
_monitor: Optional[PipelineMonitor] = None


def get_monitor(pipeline_name: str = "ml_pipeline") -> PipelineMonitor:
    """Get or create global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PipelineMonitor(pipeline_name)
    return _monitor