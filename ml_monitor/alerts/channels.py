"""
Alert delivery channels: Slack webhook, email (SMTP), generic webhook.
"""
from __future__ import annotations

import json
import logging
import smtplib
from email.mime.text import MIMEText
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)


class SlackChannel:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, model_id: str, severity: str, message: str) -> bool:
        emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")
        payload = {
            "text": f"{emoji} *ML Monitor* | `{model_id}` | {severity.upper()}\n{message}"
        }
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=5)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False


class EmailChannel:
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        recipients: List[str],
        password: str = "",
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender = sender
        self.recipients = recipients
        self.password = password

    def send(self, model_id: str, severity: str, message: str) -> bool:
        subject = f"[ML Monitor] {severity.upper()} – {model_id}"
        body = f"Model: {model_id}\nSeverity: {severity}\n\n{message}"
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as smtp:
                smtp.starttls()
                if self.password:
                    smtp.login(self.sender, self.password)
                smtp.sendmail(self.sender, self.recipients, msg.as_string())
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


class WebhookChannel:
    def __init__(self, url: str):
        self.url = url

    def send(self, model_id: str, severity: str, message: str) -> bool:
        payload = {
            "model_id": model_id,
            "severity": severity,
            "message": message,
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=5)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False