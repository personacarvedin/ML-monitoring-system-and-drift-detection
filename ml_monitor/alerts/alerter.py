"""
Alert dispatcher with cooldown tracking.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..storage.store import MonitorStore
from .channels import EmailChannel, SlackChannel, WebhookChannel

logger = logging.getLogger(__name__)


class Alerter:
    """
    Dispatches alerts to configured channels with cooldown to avoid spam.
    """

    def __init__(self, config: dict, store: MonitorStore):
        self.store = store
        alert_cfg = config.get("alerts", {})
        self.enabled = alert_cfg.get("enabled", True)
        self.cooldown = timedelta(
            minutes=alert_cfg.get("cooldown_minutes", 30)
        )

        # Track last alert times: {(model_id, alert_type) -> datetime}
        self._last_sent: Dict[tuple, datetime] = {}

        # Build channels
        self.channels = []
        channels_cfg = alert_cfg.get("channels", {})

        slack_cfg = channels_cfg.get("slack", {})
        if slack_cfg.get("enabled") and slack_cfg.get("webhook_url"):
            self.channels.append(SlackChannel(slack_cfg["webhook_url"]))

        email_cfg = channels_cfg.get("email", {})
        if email_cfg.get("enabled") and email_cfg.get("sender"):
            self.channels.append(
                EmailChannel(
                    smtp_host=email_cfg["smtp_host"],
                    smtp_port=email_cfg["smtp_port"],
                    sender=email_cfg["sender"],
                    recipients=email_cfg.get("recipients", []),
                )
            )

        webhook_cfg = channels_cfg.get("webhook", {})
        if webhook_cfg.get("enabled") and webhook_cfg.get("url"):
            self.channels.append(WebhookChannel(webhook_cfg["url"]))

        logger.info(f"Alerter initialised with {len(self.channels)} channel(s)")

    def fire(
        self,
        model_id: str,
        alert_type: str,
        message: str,
        severity: str = "warning",
    ) -> Optional[int]:
        """
        Dispatch an alert if not in cooldown.

        Returns
        -------
        alert_id if fired, else None
        """
        if not self.enabled:
            return None

        key = (model_id, alert_type)
        last = self._last_sent.get(key)
        if last and (datetime.utcnow() - last) < self.cooldown:
            logger.debug(f"Alert suppressed (cooldown): {alert_type} for {model_id}")
            return None

        self._last_sent[key] = datetime.utcnow()

        # Persist to DB
        alert_id = self.store.log_alert(
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
        )

        # Dispatch to channels
        for channel in self.channels:
            try:
                channel.send(model_id, severity, message)
            except Exception as e:
                logger.error(f"Channel dispatch error: {e}")

        logger.warning(f"ALERT [{severity.upper()}] {model_id} | {alert_type}: {message}")
        return alert_id

    def fire_drift_alert(
        self,
        model_id: str,
        feature: str,
        test_name: str,
        statistic: float,
    ) -> Optional[int]:
        message = (
            f"Drift detected on feature '{feature}' "
            f"(test={test_name}, stat={statistic:.4f})"
        )
        return self.fire(
            model_id=model_id,
            alert_type=f"drift_{feature}",
            message=message,
            severity="warning",
        )

    def fire_quality_alert(
        self, model_id: str, issues: List[str]
    ) -> Optional[int]:
        message = "Data quality issues:\n" + "\n".join(f"  • {i}" for i in issues)
        return self.fire(
            model_id=model_id,
            alert_type="data_quality",
            message=message,
            severity="warning",
        )

    def fire_performance_alert(
        self,
        model_id: str,
        metric: str,
        value: float,
        threshold: float,
    ) -> Optional[int]:
        message = (
            f"Performance degradation: {metric}={value:.4f} "
            f"(threshold={threshold:.4f})"
        )
        return self.fire(
            model_id=model_id,
            alert_type=f"perf_{metric}",
            message=message,
            severity="critical",
        )