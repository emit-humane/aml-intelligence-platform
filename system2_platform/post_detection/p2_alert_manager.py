"""
P2 — Alert Manager.

Creates, stores, and retrieves Alert objects from FusedRiskOutput.

Thresholds:
  tx_score >= 31  -> alert generated (Medium or above)
  tx_score < 31   -> silent pass-through (logged but no alert)

Storage:
  Primary  : in-memory dict (alert_id -> Alert) for fast lookups
  Secondary: SQLite via SQLAlchemy (tables auto-created on first use)
  Export   : append to data/generated_alerts.csv for every alert

Persistence behaviour:
  - ALL alerts (Medium/High/Critical) are persisted to SQLite + CSV.
  - The DB path and CSV path are configurable via constructor params.

Group risk deduplication:
  Transactions in the same community_id share a group_id tag.
  Group score = max risk score seen in that community so far.
"""

from __future__ import annotations

import csv
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..contracts.alert import Alert
from ..contracts.fused_risk_output import FusedRiskOutput, RiskLevel
from ..contracts.rule_engine_output import RuleEngineOutput
from ..contracts.behavioral_anomaly_output import BehavioralAnomalyOutput
from ..contracts.gnn_inference_output import GNNInferenceOutput
from ..contracts.live_graph_feature_vector import LiveGraphFeatureVector

# Minimum score to generate an alert
_ALERT_THRESHOLD = 31.0

_CSV_COLUMNS = [
    "alert_id", "transaction_id", "sender_account", "community_id",
    "transaction_risk_score", "group_risk_score", "risk_level",
    "risk_level_group", "triggered_patterns", "rule_explanations",
    "anomaly_drivers", "structural_anomaly_explanations",
    "explanation", "alert_status", "created_at",
]


class AlertManager:
    """
    Thread-safe alert store with SQLite + CSV persistence.

    Usage
    -----
    mgr = AlertManager()
    alert = mgr.maybe_create_alert(fused, rule_out, behav_out, gnn_out, gfv)
    alerts = mgr.get_alerts(risk_level="High")
    """

    def __init__(
        self,
        db_path: str | Path = "data/aml.db",
        csv_path: str | Path = "data/generated_alerts.csv",
    ) -> None:
        self._alerts: dict[str, Alert] = {}
        self._lock = threading.Lock()
        self._community_alerts: dict[int, list[str]] = defaultdict(list)

        self._csv_path = Path(csv_path)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path  = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Lazy SQLite init
        self._engine = None
        self._Session = None
        self._db_ready = False
        self._db_lock  = threading.Lock()

        # Ensure CSV has header
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=_CSV_COLUMNS).writeheader()

    # ------------------------------------------------------------------
    # DB init (lazy, called on first alert)
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Initialise SQLite engine + create tables if not done yet."""
        if self._db_ready:
            return
        with self._db_lock:
            if self._db_ready:
                return
            try:
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                from ..db.models import Base

                db_url = f"sqlite:///{self._db_path}"
                engine = create_engine(
                    db_url,
                    connect_args={"check_same_thread": False},
                )
                Base.metadata.create_all(bind=engine)
                self._engine  = engine
                self._Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
                self._db_ready = True
                print(f"[P2] SQLite ready: {self._db_path}")
            except Exception as exc:
                print(f"[P2] WARNING: SQLite unavailable ({exc}); using in-memory only.")
                self._db_ready = True   # prevent retries

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def maybe_create_alert(
        self,
        fused: FusedRiskOutput,
        rule_out: RuleEngineOutput,
        behav_out: BehavioralAnomalyOutput,
        gnn_out: GNNInferenceOutput,
        gfv: Optional[LiveGraphFeatureVector] = None,
    ) -> Optional[Alert]:
        """
        Create an alert if the fused score exceeds the threshold.
        Returns the Alert or None.
        """
        if fused.transaction_risk_score < _ALERT_THRESHOLD:
            return None

        community_id = gfv.sender_community_id if gfv else -1
        group_score  = self._community_max_score(community_id, fused.transaction_risk_score)

        alert = Alert(
            transaction_id=fused.transaction_id,
            sender_account=fused.sender_account,
            community_id=community_id,
            transaction_risk_score=fused.transaction_risk_score,
            group_risk_score=group_score,
            risk_level=fused.risk_level,
            risk_level_group=_risk_level(group_score),
            triggered_patterns=fused.triggered_patterns,
            rule_explanations=rule_out.rule_explanations,
            anomaly_drivers=behav_out.anomaly_drivers,
            structural_anomaly_explanations=gnn_out.structural_anomaly_explanations,
            score_breakdown=fused.score_breakdown,
            explanation=fused.explanation,
        )

        with self._lock:
            self._alerts[alert.alert_id] = alert
            self._community_alerts[community_id].append(alert.alert_id)

        # Persist async-ish in background (don't block the HTTP response)
        self._persist_alert(alert)

        return alert

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        return self._alerts.get(alert_id)

    def get_alerts(
        self,
        risk_level: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Alert]:
        with self._lock:
            alerts = list(self._alerts.values())

        if risk_level:
            alerts = [a for a in alerts if a.risk_level == risk_level]
        if status:
            alerts = [a for a in alerts if a.alert_status == status]

        alerts.sort(key=lambda a: a.created_at, reverse=True)
        return alerts[offset : offset + limit]

    def update_status(
        self, alert_id: str, status: str, assigned_to: Optional[str] = None
    ) -> Optional[Alert]:
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return None
            self._alerts[alert_id] = alert.model_copy(update={
                "alert_status": status,
                "assigned_to":  assigned_to or alert.assigned_to,
                "updated_at":   datetime.now(timezone.utc),
            })
            return self._alerts[alert_id]

    def alert_count(self) -> dict[str, int]:
        with self._lock:
            alerts = list(self._alerts.values())
        counts: dict[str, int] = {
            "total": len(alerts),
            "Low": 0, "Medium": 0, "High": 0, "Critical": 0,
            "Open": 0, "Investigating": 0, "SAR_Filed": 0, "Closed": 0,
        }
        for a in alerts:
            counts[a.risk_level] = counts.get(a.risk_level, 0) + 1
            counts[a.alert_status] = counts.get(a.alert_status, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_alert(self, alert: Alert) -> None:
        """Write alert to SQLite + CSV (best-effort, never raises)."""
        try:
            self._append_csv(alert)
        except Exception as exc:
            print(f"[P2] CSV write error: {exc}")

        try:
            self._ensure_db()
            self._write_db(alert)
        except Exception as exc:
            print(f"[P2] DB write error: {exc}")

    def _append_csv(self, alert: Alert) -> None:
        row = {
            "alert_id":             alert.alert_id,
            "transaction_id":       alert.transaction_id,
            "sender_account":       alert.sender_account,
            "community_id":         alert.community_id,
            "transaction_risk_score": alert.transaction_risk_score,
            "group_risk_score":     alert.group_risk_score,
            "risk_level":           alert.risk_level,
            "risk_level_group":     alert.risk_level_group,
            "triggered_patterns":   "|".join(alert.triggered_patterns),
            "rule_explanations":    "|".join(alert.rule_explanations),
            "anomaly_drivers":      "|".join(alert.anomaly_drivers),
            "structural_anomaly_explanations": "|".join(
                alert.structural_anomaly_explanations
            ),
            "explanation":          alert.explanation,
            "alert_status":         alert.alert_status,
            "created_at":           alert.created_at.isoformat(),
        }
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_COLUMNS).writerow(row)

    def _write_db(self, alert: Alert) -> None:
        if self._Session is None:
            return
        from ..db.models import AlertModel
        rec = AlertModel(
            alert_id=alert.alert_id,
            transaction_id=alert.transaction_id,
            sender_account=alert.sender_account,
            community_id=alert.community_id,
            transaction_risk_score=alert.transaction_risk_score,
            group_risk_score=alert.group_risk_score,
            risk_level=alert.risk_level,
            risk_level_group=alert.risk_level_group,
            triggered_patterns=alert.triggered_patterns,
            rule_explanations=alert.rule_explanations,
            anomaly_drivers=alert.anomaly_drivers,
            structural_anomaly_explanations=alert.structural_anomaly_explanations,
            score_breakdown=alert.score_breakdown,
            explanation=alert.explanation,
            alert_status=alert.alert_status,
            assigned_to=alert.assigned_to,
        )
        with self._Session() as session:
            session.add(rec)
            session.commit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _community_max_score(self, community_id: int, new_score: float) -> float:
        """Return the max risk score seen in this community (including new_score)."""
        with self._lock:
            alert_ids = self._community_alerts.get(community_id, [])
            existing_scores = [
                self._alerts[aid].transaction_risk_score
                for aid in alert_ids
                if aid in self._alerts
            ]
        if not existing_scores:
            return new_score
        return max(max(existing_scores), new_score)


def _risk_level(score: float) -> RiskLevel:
    if score <= 30:
        return "Low"
    if score <= 60:
        return "Medium"
    if score <= 80:
        return "High"
    return "Critical"
