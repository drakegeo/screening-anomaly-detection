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
│       ├── anomaly.py            ← primary detector: z-score scoring, flagging, sensitivity
│       ├── pca.py                ← PCA check (proves independence, justifies univariate)
│       ├── isoforest.py          ← Isolation Forest + SHAP (multivariate, works on independence)
│       └── visualise.py          ← all anomaly, diagnostic, and SHAP plots
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

The approach is chosen by the data, in five steps.

**Step 0 — Do the 15 series share structure? (method justification)**
Before choosing a model, we ask whether the series move together — if they did, a
joint multivariate model would be the right tool. Two independent views say no:

- **Correlation heatmap** (plot 06): pairwise correlations near zero (only ListA
  fields correlate, ~0.41).
- **PCA scree** (plot 09): the first principal component explains only **18%** of
  variance and **11 of 14 components** are needed to reach 90%. Correlated data
  would concentrate 60–90% in PC1. A PCA reconstruction detector agrees with the
  z-score detector on only **1 of 44** flags — because there is no shared
  structure to exploit.

→ **The series are independent. A *linear* multivariate model (PCA) cannot help —
it needs correlation to compress. The per-series univariate approach is therefore
the correct primary detector. (Independence does not rule out *all* multivariate
methods — see Step 3, where a tree-based model exploits exactly this independence.)**

**Step 1 — Robust Hourly Z-Score** (the primary detector) — per series, per hour-of-day:

```
z = (observed − hourly_median) / (hourly_MAD × 1.4826)
```

Flag if `z < −2.5` (drop anomaly). "Normal" is hour-of-day specific — 03:00 and
11:00 have very different baselines — so this is a per-series *time-series* model,
not a global mean. Baseline fitted on 20 weekdays, excluding weekends. Median and
MAD (not mean/std) resist the skewed, zero-heavy hit-rate distributions.

**Step 2 — Cross-field corroboration** (plot 08). A corrupted sanctions list would
depress *all* its fields at once. For each flag we check whether sibling fields of
the same list also dropped on the same date — separating **list-level failure**
(all fields) from **field-level noise** (one field).

**Step 3 — Isolation Forest + SHAP** (plots 10–12), the multivariate detector that
*works* on independent data. Where PCA fails, Isolation Forest thrives: it isolates
anomalies by randomly splitting a *single* feature at a time, so sharply-separable
independent features are isolated near the tree root. It is fed the same
standardised-residual matrix, clipped to the downward side (drops only), and scores
each hour for how easily it is isolated.

- Agreement: **11 of 44** z-score critical flags are independently caught (vs 1/44
  for PCA) — a genuine, statistically distinct second opinion.
- **SHAP reason codes** (plots 11–12) attribute each alert to specific series
  ("ListA_field1 drove this hour"). Because the features are uncorrelated, SHAP
  attributions are clean — no multicollinearity ambiguity. This is the reason code a
  compliance officer needs to act. Global SHAP importance is roughly even across all
  series, reinforcing the independence finding (no single shared driver).

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

### Conclusion — list-level vs field-level

The corroboration grid (plot 08) is the deciding evidence: **for no list do the
fields drop together on the same date.** A corrupted list would hit all its fields
simultaneously — that signature is absent everywhere.

→ **No list-level failure occurred in this window.** Every real anomaly is
field-scoped (a data-pipeline / configuration issue on one field), not sanctions
list corruption. Operationally reassuring: the screening lists appear healthy.

**Event-level counting:** 57 hour-level flags collapse to **42 events (17 non-holiday)** —
a sustained multi-hour drop is one operational event, not many alerts (see `outputs/events.csv`).

The field-level drops worth investigating (ranked by the event view):
- **ListC_field9** — the top offender: worst z=−4.30, 5 events, solid baseline (median 0.09).
- **ListA_field3** — the clearest *systematic* signature: recurring drops at exactly 10:00 on
  Jan 2, 3, 4, 5 (same field, same hour, consecutive days) — stronger evidence than raw severity.
- **ListD_field11** — recurring drops (Dec 29, Jan 12).

**Every top alert rests on a full 20-observation baseline** (zero low-confidence active cells
across all series — see `outputs/alert_confidence_audit.csv`), so none is a low-volume artifact.

Holiday flags (Dec 25 / Jan 1) are volume effects, not failures.

**Recommended alerting rule:** escalate only when ≥2 fields of the same list drop
together (list-level); log single-field drops for review. Under this rule no
false alarm would have fired here — the correct outcome.

---

## Validation

With no ground-truth labels, the detector is validated by **synthetic drop injection**
(`python -m pipelines.validation`, plot 13). Known drops are injected analytically into each
baseline cell and scored with the production formula:

```
obs_injected = hourly_median × (1 − drop_fraction);  detected if z < −2.5
```

Recovery curve across all 128 active cells: catches large failures reliably (59% at 90% drop),
weaker on subtle ones (21% at 50% drop) — **calibrated for precision (0.25% false-alarm rate),
not recall.** The overnight blind spots on the coverage heatmap are the hidden-denominator
problem made visible: low volume → wide MAD → only large drops clear the threshold.

**Design choices tested, not assumed:**
- **Hour-of-day vs hour × day-of-week:** day-of-week reduces residual spread only 11.5% while
  cutting baseline from 20 → 4 obs/cell. Hour-only is the right granularity.

---

## Known Limitations

1. **Hidden denominator** — only hit rate ratio available, not underlying request volume. Cannot distinguish genuine drop from low-volume noise. **The single most valuable upgrade: log request counts.**
2. **No ground truth** — validated by synthetic injection (analytical recovery curve), not precision/recall against labels.
3. **Small baseline** — 20 observations per hour × series. Estimates have uncertainty at sparse hours.
4. **Boiling-frog risk** — a rolling baseline would adapt to slow degradation and hide it. Production fix: freeze a golden baseline from a certified-healthy period.
5. **Weekend exclusion** — assumed correct per brief; weekend behaviour unknown.

---

*Data is proprietary Swift operational data provided under confidentiality for this hiring exercise.*
