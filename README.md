# Screening Hit Rate Anomaly Detection
**Swift Senior Applied Data Scientist — Take-Home Case**

Detects drop anomalies in sanctions screening hit rate data. A drop anomaly is a value that falls significantly below the expected hourly pattern for a specific list/field combination — an early-warning signal for corrupted or truncated sanctions lists.

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e .
```

---

## How to Run

**Step 1 — Data processing + EDA plots:**
```bash
python -m pipelines.processing
```
Outputs: `outputs/figures/01_*.png` through `06_*.png`, `outputs/data_quality_report.csv`

**Step 2 — Anomaly detection:**
```bash
python -m pipelines.detection
```
Outputs: `outputs/figures/07_threshold_sensitivity.png`, `outputs/figures/anomaly_*.png`, `outputs/anomalies.csv`

---

## Project Structure

```
screening-anomaly-detection/
│
├── data/
│   └── screening_hitrate.csv     ← proprietary Swift data (not committed)
│
├── src/
│   ├── processing/
│   │   ├── load.py               ← data loading, validation, holiday tagging
│   │   └── eda.py                ← 6 EDA plots
│   └── detection/
│       ├── baseline.py           ← robust hourly baseline (median + MAD per series × hour)
│       ├── anomaly.py            ← scoring, flagging, threshold sensitivity
│       └── visualise.py          ← per-series anomaly plots + overview
│
├── pipelines/
│   ├── processing.py             ← entry point: EDA
│   └── detection.py              ← entry point: anomaly detection
│
├── outputs/
│   ├── figures/                  ← all saved plots
│   └── anomalies.csv             ← flagged anomalies table
│
├── notebooks/
│   └── 01_exploration.ipynb      ← exploratory analysis
│
├── pyproject.toml
├── ai_usage_log.md
└── CLAUDE.md                     ← project specification
```

---

## Methodology

**Robust Hourly Z-Score** — per series, per hour-of-day:

```
z = (observed − hourly_median) / (hourly_MAD × 1.4826)
```

Flag if `z < −2.5` (drop anomaly). Baseline fitted on 20 weekdays, excluding weekends. Uses median and MAD instead of mean/std to resist skewed, zero-heavy distributions.

**Three flag types:**

| Symbol | Type | Meaning |
|--------|------|---------|
| `○` | `drop_zscore` | Statistically significant drop below hourly baseline |
| `○` | `contextual_zero` | Zero during a normally active hour (non-holiday) |
| `×` | `contextual_zero_holiday` | Zero on Dec 25 / Jan 1 — volume effect, not list failure |

**Special handling:**
- `ListB_field5` — excluded (inactive, always zero, no baseline possible)
- `ListB_field6` — wider threshold (−2.0), excluded from contextual zero (high missingness, unstable)
- `NaN` values — flagged as data collection failures, not anomalies

---

## Results Summary (threshold = −2.5)

| Series | Critical flags | Holiday zeros | Worst z |
|--------|---------------|---------------|---------|
| ListA_field1 | 2 | 0 | −6.56 |
| ListA_field3 | 8 | 6 | −4.31 |
| ListC_field9 | 10 | 0 | −4.31 |
| ListE_field13 | 5 | 2 | −3.79 |
| ListD_field10 | 9 | 4 | −3.70 |
| ListB_field4 | 7 | 0 | −3.66 |
| ListD_field11 | 3 | 0 | −3.25 |
| ListE_field14 | 0 | 1 | — |

17 non-holiday critical flags across 7 series. Most holiday-period flags are volume-driven.

---

## Known Limitations

1. **Hidden denominator** — only hit rate ratio available, not underlying request volume. Cannot distinguish genuine drop from low-volume noise.
2. **No ground truth** — validation is qualitative and visual; no precision/recall computable.
3. **Small baseline** — 20 observations per hour × series. Estimates have uncertainty at sparse hours.
4. **Weekend exclusion** — assumed correct per brief; weekend behaviour unknown.

---

*Data is proprietary Swift operational data provided under confidentiality for this hiring exercise.*
