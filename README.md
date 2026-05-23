# ML Monitoring and Drift Detection

## Overview

This is a Python-based production monitoring system designed to detect when machine learning models begin to fail silently in the real world. Once a model is deployed, the data it receives in production rarely stays the same as the data it was trained on — distributions shift, new patterns emerge, and performance degrades without any obvious error being thrown. ML Monitor solves this by continuously comparing live production data against a reference baseline, flagging statistical drift, tracking model performance over time, and sending alerts before degradation becomes a business problem.

The system is built to be lightweight and self-contained, requiring no external infrastructure. Everything runs locally — a SQLite database stores all monitoring history, a Streamlit dashboard visualises it, and alerts can be routed to Slack, email, or any webhook endpoint. It is designed for data scientists and ML engineers who want observability into their deployed models without the overhead of a large monitoring platform.

---

## Why This Is Required

A trained machine learning model is essentially a snapshot of the world at the time its training data was collected. The moment that model goes live, the world keeps changing. Customer behaviour shifts. Data pipelines change upstream. Seasonal patterns kick in. Sensor calibration drifts. Any of these can quietly erode model performance — often without raising a single exception or error log.

This problem is known as **model drift**, and it is one of the most common and costly failure modes in production ML. Without monitoring:

- A classification model can slip from 92% accuracy to 74% over six weeks with nobody noticing until a downstream business metric suffers.
- A feature that drove predictions can shift entirely — for example, income levels in a credit scoring model — making the model's learned weights meaningless for the new distribution.
- Data quality issues such as null values, schema changes, or outliers introduced by upstream pipeline changes silently corrupt predictions.

ML Monitor exists to make these failures visible and actionable before they cause damage.

---

## Goals and Objectives

**Primary goal:** Provide a complete, production-ready monitoring layer that can be dropped into any ML workflow with minimal setup.

**Objectives:**

- Detect statistical drift in input features as early as possible using multiple complementary tests.
- Track model performance metrics over time and alert when they fall below defined thresholds.
- Validate incoming data quality on every production batch, catching schema violations, null rates, and outliers.
- Persist all monitoring history in a queryable store so trends can be analysed retrospectively.
- Surface everything through an interactive dashboard that requires no SQL knowledge to use.
- Keep the system dependency-light so it runs in any Python environment, including Codespaces, local machines, and cloud VMs.

---

## What the Project Does

ML Monitor wraps around your existing model and data pipeline. You register a model once, providing your training data as the reference baseline. Then on every incoming production batch, you call a single function. Internally, three independent services run in sequence.

**Drift Detection** compares the distribution of each feature in the production batch against the reference distribution using statistical hypothesis tests. For continuous numerical features it runs both the Kolmogorov-Smirnov test and the Population Stability Index. The KS test is sensitive to any distributional difference, while PSI specifically measures how much a population has shifted — a metric widely used in financial risk modelling. For categorical features it uses the Chi-square goodness-of-fit test to detect changes in category proportions. For high-dimensional data such as embeddings, it uses Maximum Mean Discrepancy with an RBF kernel, which measures distributional distance in a kernel-defined feature space and uses permutation testing to compute an empirical p-value.

**Performance Monitoring** computes model accuracy metrics against ground truth labels whenever they are available. For classifiers this includes accuracy, precision, recall, F1 score, and ROC-AUC. For regressors it computes MSE, RMSE, MAE, and R². Each metric is compared against thresholds you define at registration time, and a critical alert fires if any metric falls below its threshold.

**Data Quality Checking** runs independently of drift detection and catches issues that statistical tests would not flag. It computes the null rate for every column and flags any that exceed a configurable threshold. It detects outliers using z-score relative to the reference distribution. It validates the schema of the incoming data against the expected column names and data types, catching upstream pipeline changes that would corrupt predictions silently.

Everything is logged to SQLite with a timestamp. The alerter runs after all three services complete and dispatches notifications through whichever channels you have configured, with a per-alert-type cooldown window to prevent notification spam.

---

## Project Structure

```text
ml_monitor/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── ml_monitor/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── monitor.py          # Main monitoring orchestrator
│   │   └── registry.py         # Model registry
│   ├── drift/
│   │   ├── __init__.py
│   │   ├── detector.py         # Drift detection engine
│   │   ├── statistical.py      # KS, PSI, chi-square tests
│   │   └── embeddings.py       # Embedding drift (MMD)
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── performance.py      # Accuracy, F1, AUC, RMSE etc.
│   │   └── data_quality.py     # Nulls, outliers, schema checks
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── alerter.py          # Alert dispatcher
│   │   └── channels.py         # Slack / email / webhook
│   ├── storage/
│   │   ├── __init__.py
│   │   └── store.py            # Time-series log store (SQLite)
│   └── dashboard/
│       ├── __init__.py
│       └── app.py              # Streamlit dashboard
├── examples/
│   ├── train_baseline.py
│   └── simulate_drift.py
└── tests/
    ├── test_drift.py
    └── test_metrics.py
```

## System Flow

```
Your production batch (pandas DataFrame)
               │
               ▼
        MLMonitor.run()
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
  Drift     Metrics    Data
Detection   Tracking   Quality
    │          │          │
    └──────────┴──────────┘
               │
               ▼
          Alerter
    (Slack / Email / Webhook)
               │
               ▼
         SQLite Database
               │
               ▼
      Streamlit Dashboard
```

---

## The Data

### Why Synthetic Data

The project ships with two simulation scripts that generate synthetic monitoring history rather than relying on a real dataset. This is intentional. Real production monitoring data is proprietary and context-specific — a credit scoring model's drift patterns look nothing like a demand forecasting model's. Synthetic data lets the system demonstrate all of its detection capabilities in a controlled, reproducible way that anyone can run immediately without needing access to proprietary systems.

### How the Data Is Generated

`train_baseline.py` uses scikit-learn's `make_classification` utility to generate a 3,000-sample binary classification dataset with 10 features, 6 of which are informative. A Random Forest classifier is trained on 70% of this data. The remaining 30% forms the reference baseline stored in the model registry.

The script then simulates 30 consecutive days of production batches. For the first 15 days the production data is drawn from the same distribution as the training data — clean, no drift. From day 15 onward, features 0 and 1 are progressively shifted by an increasing offset, simulating gradual covariate drift. From day 20 onward, a proportion of the model's predictions are randomly flipped, simulating performance degradation as the model struggles with the shifted distribution. Each day's batch is logged to SQLite and then back-dated so the dashboard shows a realistic 30-day history.

`simulate_drift.py` registers a second model (`drift_demo`) with a separate reference baseline and simulates three distinct drift patterns across 30 days. Feature 0 drifts linearly — a steady, predictable slide. Feature 1 drifts exponentially — slowly at first then rapidly accelerating. Feature 2 is completely clean for 20 days and then jumps suddenly on day 21, simulating a one-time upstream pipeline change. This variety ensures the heatmap and drill-down charts show meaningfully different shapes across features.

### What Is Being Stored

Every monitoring run writes to three tables in `ml_monitor.db`.

`drift_records` stores one row per feature per statistical test per run. Each row records the test name, the computed statistic, the p-value where applicable, and a boolean flag indicating whether drift was detected at the configured threshold.

`metric_records` stores one row per metric per run. Accuracy, F1, AUC, and any other computed metrics are each stored as a separate row with their value and timestamp.

`alert_records` stores one row per fired alert. Each alert has a type, severity level, a human-readable message describing what triggered it, and a resolved flag that can be set once the issue is addressed.

---

## Installation

Clone the repository and set up the environment (Mac and Linux):

```bash
git clone https://github.com/personacarvedin/ML-monitoring-system-and-drift-detection.git
cd ML-monitoring-system-and-drift-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Generate baseline monitoring history

Trains the model and simulates 30 days of production data with gradual drift and performance degradation.

```bash
python examples/train_baseline.py
```

![Train Baseline](https://raw.githubusercontent.com/personacarvedin/ML-monitoring-system-and-drift-detection/main/Screenshots/Train_baseline.png)

---

### Step 2 — Generate drift simulation

Registers the drift demo model and simulates 30 days of three distinct drift patterns.

```bash
python examples/simulate_drift.py
```

![Simulate Drift](https://raw.githubusercontent.com/personacarvedin/ML-monitoring-system-and-drift-detection/main/Screenshots/Simulate_drift.png)

---

### Step 3 — Inspect the database

Connects to `ml_monitor.db` and prints the first 5 rows of every table so you can verify data was written correctly before opening the dashboard.

```bash
python data_showing.py
```

This will show the three tables — `drift_records`, `metric_records`, and `alert_records` — with their columns and sample rows, confirming that drift flags, metric values, and alert messages are all being stored as expected.

![SQL Screenshot](https://raw.githubusercontent.com/personacarvedin/ML-monitoring-system-and-drift-detection/main/Screenshots/SQL.png)

---

### Step 4 — Launch the dashboard

```bash
streamlit run ml_monitor/dashboard/app.py
```

Open the URL printed in the terminal. In GitHub Codespaces, click the "Open in Browser" popup or go to the Ports tab and open port 8501.

The dashboard has five sections:

**Snapshot** — four KPI cards at the top summarising total drift tests run, how many flagged drift, how many alerts are open, and the latest metric value.

**Alerts** — tabbed view of open vs all alerts with severity colour coding.

**Performance Metrics** — latest value cards for each metric and a trend line chart showing how they evolved over the 30-day window.

**Drift Detection** — a drift rate bar chart showing which features drifted most frequently, a heatmap plotting drift across features and days (green = clean, red = drift detected), and a raw records table.

**Feature Drill-down** — select any individual feature to see its test statistic and p-value plotted over time, with the significance threshold drawn as a reference line.

![Dashboard Screenshot](https://raw.githubusercontent.com/personacarvedin/ML-monitoring-system-and-drift-detection/main/Screenshots/Dashboard.png)

![Heatmap Screenshot](https://raw.githubusercontent.com/personacarvedin/ML-monitoring-system-and-drift-detection/main/Screenshots/Heatmap.png)

![Feature Drilldown Screenshot](https://raw.githubusercontent.com/personacarvedin/ML-monitoring-system-and-drift-detection/main/Screenshots/feature_drilldown.png)

---

## Resetting and Regenerating Data

To wipe all monitoring history and start fresh:

```bash
rm ml_monitor.db
python examples/train_baseline.py
python examples/simulate_drift.py
```

The database is recreated automatically on the next run.

---

## Configuration

All thresholds and settings are in `config/config.yaml`. No code changes are needed to tune the system's sensitivity or enable alert channels.

Key settings:

- `drift.ks_threshold` — p-value cutoff for the KS test. Lower values make the test more conservative.
- `drift.psi_threshold` — PSI cutoff. The industry standard is 0.2 for significant drift.
- `alerts.cooldown_minutes` — minimum gap between repeat alerts for the same issue, to prevent notification spam.
- `storage.retention_days` — how many days of history to keep before old records are purged.
- Alert channels (Slack, email, webhook) are each independently toggled with `enabled: true/false`.
