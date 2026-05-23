"""
Generates 30 days of monitoring history for credit_risk_v1.
Each iteration simulates a new production batch arriving daily,
with gradually degrading performance to make the charts interesting.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_monitor.core.monitor import MLMonitor
from sqlalchemy import text

N_DAYS = 30
FEATURES = [f"feat_{i}" for i in range(10)]

print("Training baseline model...")
X, y = make_classification(n_samples=3000, n_features=10, n_informative=6,
                            n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

monitor = MLMonitor(config_path="config/config.yaml")
ref_df = pd.DataFrame(X_train, columns=FEATURES)

monitor.register_model(
    model_id="credit_risk_v1",
    task="classification",
    reference_data=ref_df,
    performance_thresholds={"accuracy": 0.80, "f1": 0.75},
)

print(f"Simulating {N_DAYS} days of production data...\n")
base_time = datetime.utcnow() - timedelta(days=N_DAYS)

for day in range(N_DAYS):
    rng = np.random.default_rng(day + 100)
    shift = max(0.0, (day - 15) * 0.18) if day >= 15 else 0.0

    X_prod, y_prod = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_redundant=2, random_state=day + 200
    )
    X_prod = X_prod.astype(float)
    X_prod[:, 0] += shift
    X_prod[:, 1] += shift * 0.5

    prod_df = pd.DataFrame(X_prod, columns=FEATURES)
    y_pred  = clf.predict(X_prod)
    y_proba = clf.predict_proba(X_prod)

    if day >= 20:
        n_flip = int(len(y_pred) * (day - 20) * 0.02)
        flip_idx = rng.choice(len(y_pred), size=min(n_flip, len(y_pred)//2), replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]

    # Disable alerter cooldown so all days log properly
    monitor.alerter.cooldown = timedelta(seconds=0)

    report = monitor.run(
        model_id="credit_risk_v1",
        production_data=prod_df,
        y_true=y_prod,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    # Back-date all freshly inserted records to this day
    record_time = (base_time + timedelta(days=day)).strftime("%Y-%m-%d %H:%M:%S")
    with monitor.store.engine.connect() as conn:
        conn.execute(text(
            "UPDATE drift_records SET timestamp=:ts WHERE model_id='credit_risk_v1' "
            "AND timestamp=(SELECT MAX(timestamp) FROM drift_records WHERE model_id='credit_risk_v1')"
        ), {"ts": record_time})
        conn.execute(text(
            "UPDATE metric_records SET timestamp=:ts WHERE model_id='credit_risk_v1' "
            "AND timestamp=(SELECT MAX(timestamp) FROM metric_records WHERE model_id='credit_risk_v1')"
        ), {"ts": record_time})
        conn.commit()

    perf = report["sections"].get("performance", {})
    print(f"Day {day+1:02d} | drift_shift={shift:.2f} | accuracy={perf.get('accuracy', 0):.3f} | f1={perf.get('f1', 0):.3f}")

print("\nDone! Run: streamlit run ml_monitor/dashboard/app.py")