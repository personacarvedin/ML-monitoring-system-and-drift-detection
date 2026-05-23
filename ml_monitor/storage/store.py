"""
Time-series storage for monitoring logs using SQLite via SQLAlchemy.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import (Column, DateTime, Float, Integer, String, Text,
                        create_engine, text)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DriftRecord(Base):
    __tablename__ = "drift_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_id = Column(String(256), index=True)
    feature = Column(String(256))
    test_name = Column(String(64))
    statistic = Column(Float)
    p_value = Column(Float, nullable=True)
    drift_detected = Column(Integer)   # 0 or 1 (SQLite has no bool)
    metadata_ = Column("metadata", Text, default="{}")


class MetricRecord(Base):
    __tablename__ = "metric_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_id = Column(String(256), index=True)
    metric_name = Column(String(128))
    value = Column(Float)
    metadata_ = Column("metadata", Text, default="{}")


class AlertRecord(Base):
    __tablename__ = "alert_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    model_id = Column(String(256), index=True)
    alert_type = Column(String(64))
    severity = Column(String(16))
    message = Column(Text)
    resolved = Column(Integer, default=0)


class MonitorStore:
    def __init__(self, db_path: str = "ml_monitor.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Store initialised at {db_path}")

    # ------------------------------------------------------------------ #
    # Drift
    # ------------------------------------------------------------------ #
    def log_drift(
        self,
        model_id: str,
        feature: str,
        test_name: str,
        statistic: float,
        drift_detected: bool,
        p_value: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        with Session(self.engine) as session:
            record = DriftRecord(
                model_id=model_id,
                feature=feature,
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                drift_detected=int(drift_detected),
                metadata_=json.dumps(metadata or {}),
            )
            session.add(record)
            session.commit()

    def get_drift_history(
        self,
        model_id: str,
        days: int = 7,
        feature: Optional[str] = None,
    ) -> List[Dict]:
        since = datetime.utcnow() - timedelta(days=days)
        with Session(self.engine) as session:
            q = session.query(DriftRecord).filter(
                DriftRecord.model_id == model_id,
                DriftRecord.timestamp >= since,
            )
            if feature:
                q = q.filter(DriftRecord.feature == feature)
            rows = q.order_by(DriftRecord.timestamp).all()
        return [self._drift_to_dict(r) for r in rows]

    def _drift_to_dict(self, r: DriftRecord) -> Dict:
        return {
            "timestamp": r.timestamp.isoformat(),
            "model_id": r.model_id,
            "feature": r.feature,
            "test_name": r.test_name,
            "statistic": r.statistic,
            "p_value": r.p_value,
            "drift_detected": bool(r.drift_detected),
            "metadata": json.loads(r.metadata_),
        }

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def log_metric(
        self,
        model_id: str,
        metric_name: str,
        value: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        with Session(self.engine) as session:
            record = MetricRecord(
                model_id=model_id,
                metric_name=metric_name,
                value=value,
                metadata_=json.dumps(metadata or {}),
            )
            session.add(record)
            session.commit()

    def get_metric_history(
        self,
        model_id: str,
        metric_name: str,
        days: int = 7,
    ) -> List[Dict]:
        since = datetime.utcnow() - timedelta(days=days)
        with Session(self.engine) as session:
            rows = (
                session.query(MetricRecord)
                .filter(
                    MetricRecord.model_id == model_id,
                    MetricRecord.metric_name == metric_name,
                    MetricRecord.timestamp >= since,
                )
                .order_by(MetricRecord.timestamp)
                .all()
            )
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "metric_name": r.metric_name,
                "value": r.value,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Alerts
    # ------------------------------------------------------------------ #
    def log_alert(
        self,
        model_id: str,
        alert_type: str,
        severity: str,
        message: str,
    ) -> int:
        with Session(self.engine) as session:
            record = AlertRecord(
                model_id=model_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    def resolve_alert(self, alert_id: int) -> None:
        with Session(self.engine) as session:
            session.execute(
                text("UPDATE alert_records SET resolved=1 WHERE id=:id"),
                {"id": alert_id},
            )
            session.commit()

    def get_open_alerts(self, model_id: Optional[str] = None) -> List[Dict]:
        with Session(self.engine) as session:
            q = session.query(AlertRecord).filter(AlertRecord.resolved == 0)
            if model_id:
                q = q.filter(AlertRecord.model_id == model_id)
            rows = q.order_by(AlertRecord.timestamp.desc()).all()
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "model_id": r.model_id,
                "alert_type": r.alert_type,
                "severity": r.severity,
                "message": r.message,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Purge old data
    # ------------------------------------------------------------------ #
    def purge_old_records(self, retention_days: int = 90) -> None:
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        with Session(self.engine) as session:
            for model in (DriftRecord, MetricRecord, AlertRecord):
                session.execute(
                    text(f"DELETE FROM {model.__tablename__} WHERE timestamp < :cutoff"),
                    {"cutoff": cutoff},
                )
            session.commit()
        logger.info(f"Purged records older than {retention_days} days")