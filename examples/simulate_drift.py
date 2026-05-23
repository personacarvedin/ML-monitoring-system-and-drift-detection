"""
Generates 30 days of escalating drift for drift_demo model.
Features 0-2 drift progressively. Makes heatmap and bar charts rich.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import make_classification

from ml_monitor.core.monitor import MLMonitor
from sqlalchemy import text

N_DAYS = 30
FEATURES = [f"feat_{i}" for i in range(10)]

np.random.seed(0)
X_ref, _ = make_classification(n_samples=1500, n_features=10, random_state=0)
ref_df = pd.DataFrame(X_ref, columns=FEATURES)

monitor = MLMonitor(config_path="config/config.yaml")
monitor.register_model(
    model_id="drift_demo",
    task="classification",
    reference_data=ref_df,
)
monitor.alerter.cooldown = timedelta(seconds=0)

print(f"Simulating {N_DAYS} days of escalating drift...\n")
base_time = datetime.utcnow() - timedelta(days=N_DAYS)

for day in range(N_DAYS):
    # Escalating drift on first 3 features
    shift_0 = day * 0.12          # feat_0: steady climb
    shift_1 = (day ** 1.3) * 0.04 # feat_1: accelerating
    shift_2 = 2.0 if day >= 20 else 0.0  # feat_2: sudden jump at day 20

    X_prod, _ = make_classification(
        n_samples=400, n_features=10, random_state=day + 500
    )
    X_prod = X_prod.astype(float)
    X_prod[:, 0] += shift_0
    X_prod[:, 1] += shift_1
    X_prod[:, 2] += shift_2

    prod_df = pd.DataFrame(X_prod, columns=FEATURES)

    report = monitor.run(model_id="drift_demo", production_data=prod_df)

    record_time = (base_time + timedelta(days=day)).strftime("%Y-%m-%d %H:%M:%S")
    with monitor.store.engine.connect() as conn:
        conn.execute(text(
            "UPDATE drift_records SET timestamp=:ts WHERE model_id='drift_demo' "
            "AND timestamp=(SELECT MAX(timestamp) FROM drift_records WHERE model_id='drift_demo')"
        ), {"ts": record_time})
        conn.commit()

    drift_rows = report["sections"].get("drift", [])
    flagged = [r["feature"] for r in drift_rows if isinstance(r, dict) and r.get("drift")]
    print(f"Day {day+1:02d} | shifts=({shift_0:.1f}, {shift_1:.1f}, {shift_2:.1f}) | drifted={flagged}")

print("\nDone! Run: streamlit run ml_monitor/dashboard/app.py")