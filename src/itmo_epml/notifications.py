"""Notification system for pipeline results."""

import json
import logging
import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for pipeline execution results."""

    pipeline_name: str
    status: str  # "success" or "failed"
    duration_seconds: float
    metrics: dict
    error: Optional[str] = None
    run_id: Optional[str] = None


class NotificationChannel(ABC):
    """Base class for notification channels."""

    @abstractmethod
    def send(self, result: PipelineResult) -> bool:
        """Send notification. Returns True if successful."""
        pass


class ConsoleNotification(NotificationChannel):
    """Print notifications to console."""

    def send(self, result: PipelineResult) -> bool:
        status_emoji = "✅" if result.status == "success" else "❌"

        message = f"""
{status_emoji} Pipeline '{result.pipeline_name}' {result.status.upper()}

Duration: {result.duration_seconds:.2f} seconds
Run ID: {result.run_id or 'N/A'}

Metrics:
{json.dumps(result.metrics, indent=2)}
"""
        if result.error:
            message += f"\nError: {result.error}"

        print(message)
        return True


class FileNotification(NotificationChannel):
    """Save notifications to file."""

    def __init__(self, output_dir: str = "reports/notifications"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def send(self, result: PipelineResult) -> bool:
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"notification_{result.pipeline_name}_{timestamp}.json"

            data = {
                "pipeline_name": result.pipeline_name,
                "status": result.status,
                "duration_seconds": result.duration_seconds,
                "metrics": result.metrics,
                "error": result.error,
                "run_id": result.run_id,
                "timestamp": datetime.now().isoformat(),
            }

            with open(self.output_dir / filename, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Notification saved to {self.output_dir / filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save notification: {e}")
            return False


class EmailNotification(NotificationChannel):
    """Send notifications via email."""

    def __init__(
        self,
        smtp_host: str = None,
        smtp_port: int = 587,
        username: str = None,
        password: str = None,
        from_addr: str = None,
        to_addrs: list = None
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port
        self.username = username or os.getenv("SMTP_USERNAME")
        self.password = password or os.getenv("SMTP_PASSWORD")
        self.from_addr = from_addr or os.getenv("NOTIFICATION_FROM")
        self.to_addrs = to_addrs or os.getenv("NOTIFICATION_TO", "").split(",")

    def send(self, result: PipelineResult) -> bool:
        if not all([self.smtp_host, self.username, self.password, self.from_addr, self.to_addrs]):
            logger.warning("Email configuration incomplete")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{result.status.upper()}] Pipeline: {result.pipeline_name}"
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            # Plain text version
            text = f"""
Pipeline: {result.pipeline_name}
Status: {result.status.upper()}
Duration: {result.duration_seconds:.2f} seconds
Run ID: {result.run_id or 'N/A'}

Metrics:
{json.dumps(result.metrics, indent=2)}

Error: {result.error or 'None'}
"""

            # HTML version
            status_color = "#28a745" if result.status == "success" else "#dc3545"
            html = f"""
<html>
<body>
<h2 style="color: {status_color}">Pipeline: {result.pipeline_name}</h2>
<p><strong>Status:</strong> {result.status.upper()}</p>
<p><strong>Duration:</strong> {result.duration_seconds:.2f} seconds</p>
<p><strong>Run ID:</strong> {result.run_id or 'N/A'}</p>
<h3>Metrics:</h3>
<pre>{json.dumps(result.metrics, indent=2)}</pre>
</body>
</html>
"""

            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            logger.info("Email notification sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


class NotificationManager:
    """Manage multiple notification channels."""

    def __init__(self):
        self.channels: list[NotificationChannel] = []

    def add_channel(self, channel: NotificationChannel) -> "NotificationManager":
        """Add notification channel."""
        self.channels.append(channel)
        return self

    def notify(self, result: PipelineResult) -> dict[str, bool]:
        """Send notification to all channels."""
        results = {}
        for channel in self.channels:
            channel_name = channel.__class__.__name__
            try:
                results[channel_name] = channel.send(result)
            except Exception as e:
                logger.error(f"Notification channel {channel_name} failed: {e}")
                results[channel_name] = False
        return results


def create_default_notifier() -> NotificationManager:
    """Create notifier with default channels."""
    return (
        NotificationManager()
        .add_channel(ConsoleNotification())
        .add_channel(FileNotification())
    )