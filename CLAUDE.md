# CLAUDE.md — Screening Hit Rate Anomaly Detection
## Swift Senior Applied Data Scientist — Take-Home Case

---

## Project Goal

Build a **clean, explainable, robust anomaly detection pipeline** that identifies
**drop anomalies** in sanctions screening hit rate data. A drop anomaly is a value
that falls significantly below what is expected given the normal hourly pattern for
that specific list/field combination.

The output should be:
- A well-structured Python codebase (modular, documented, reproducible)
- Clear visualisations that can be directly used in a PowerPoint presentation
- An honest, rigorous write-up of methodology, assumptions, and limitations

**Guiding principle: clever and defensible beats complex and opaque.**

---

## Business Context

Financial institutions are legally required to screen every transaction against
sanctions lists (OFAC, UN, EU, national lists) before allowing it to proceed.
Swift's automated screening systems do this using name-matching (fuzzy) algorithms.

**The risk this project addresses:** If a sanctions list gets updated incorrectly
(truncated, corrupted, missing entries), its hit rate drops silently — transactions
that should be flagged sail through undetected. Monitoring hit rate drop anomalies
is an early-warning system for this exact failure mode.

**Why drops matter more than spikes:** A spike means more transactions get reviewed
(noisy but safe). A drop means the list is no longer catching what it should
(silent and dangerous — potential regulatory breach affecting thousands of messages).

---

## Understanding the Data

### File
`screening_hitrate.csv`

### Structure
- **480 rows** — one per hour, covering **20 weekdays** (weekends excluded)
- **Date range:** 2023-12-18 to 2024-01-12 UTC
- **15 columns** — one per List × Field combination
- **3 weekend gaps** visible as 2-day jumps in the timestamp index

### The 15 Series
5 sanctions lists × 3 fields each:

| List  | Fields            |
|-------|-------------------|
| ListA | field1, field2, field3  |
| ListB | field4, field5, field6  |
| ListC | field7, field8, field9  |
| ListD | field10, field11, field12 |
| ListE | field13, field14, field15 |

Each list plausibly represents a distinct sanctions regime (OFAC, UN, EU, UK, national).
Each field plausibly represents a different data element of the Swift payment message
being screened (e.g. ordering customer name, beneficiary name, institution BIC).
The exact identities are anonymised — treat each series as independent.

### The Metric: Hit Rate

```
Hit Rate (List L, Field F, Hour H) = Total hits generated / Total screening requests processed
```

Where:
- **Denominator** = number of screening requests for that List × Field in that hour
  (i.e. how many transaction messages flowed through and had that field populated)
- **Numerator** = total fuzzy-match alerts generated across all those requests
- **One request can generate multiple hits** (one name can match multiple sanctions
  list entries via fuzzy matching — aliases, transliterations, similar entries)
- Therefore **hit rate > 1 is valid**: e.g. 1,500 hits / 1,000 requests = 1.5
- **Theoretical maximum** = size of the sanctions list (if every request matched
  every entry — never observed in practice, values stay well below 1 for most series)

### Key Data Quality Facts (verified from the data)

| Series        | Notable characteristic |
|---------------|------------------------|
| ListB_field5  | Active only 1/480 hours — effectively a dead/inactive series |
| ListB_field6  | 145/480 hours missing (30% missing) — chronic data collection issue; 5 hours with values > 1 (max = 2.0) |
| ListA_field1  | Zero 223/480 hours — field only populated in certain message types |
| ListB_field4  | Never zero, consistent baseline — most stable series in dataset |
| ListC_field7  | Zero 307/480 hours — sparse field type |

### What zero means vs what NaN means
- **Zero (0.0):** No screening requests arrived for that List × Field in that hour.
  This is **not an anomaly** — it reflects low/no transaction volume (e.g. overnight).
- **NaN / missing:** Data collection from the screening system failed for that hour.
  This is a **separate operational issue** — treat as missing, not as a drop.

### The Hidden Denominator Problem
The dataset contains only the ratio (hit rate), not the underlying request volume.
This means:
- We cannot directly distinguish "genuine drop" from "low-volume noise"
- Overnight hours (low volume) will naturally produce noisier, less reliable hit rates
- The same hit rate value of 0.05 means something very different when derived from
  50,000 requests vs 3 requests
- **Mitigation:** Use hour-of-day to proxy volume — business hours = high volume =
  tighter anomaly bands; overnight = low volume = wider bands or exclusion

### Correlation Structure
- Within-list correlation (do the 3 fields of the same list move together?):
  Only **ListA** shows meaningful within-list correlation (avg ~0.41).
  Other lists show near-zero correlation across their fields.
- Cross-list field-position correlation: weak across all groups (~0.08–0.17)
- **Conclusion:** Treat all 15 series as **independent** for anomaly detection.
  Do not assume shared structure. For any flagged anomaly, check post-hoc whether
  other fields of the same list also dipped (corroborating evidence, not assumption).

---

## What We Want to Build

### Philosophy
- **Per-series, per-hour-of-day baseline** — "normal" for ListA_field1 at 03:00 is
  very different from "normal" at 11:00. Anomaly = deviation from the expected
  value for that specific hour, not from a global mean.
- **Robust statistics** — use median and MAD (Median Absolute Deviation) rather
  than mean and std, because hit rate distributions are skewed and zero-heavy.
- **Drop-only detection** — we only care about values significantly below baseline.
  Upward spikes are not the target (and may reflect legitimate list updates adding entries).
- **Explainability over complexity** — a well-justified statistical approach that a
  compliance officer can understand is more valuable than a black-box ML model.

### Recommended Approach: Robust Hourly Z-Score

For each series and each hour-of-day (0–23), fit a baseline from the 20 available
data points (one per weekday in the dataset). Then score each observation:

```
Anomaly Score = (observed value - hourly median) / (hourly MAD × 1.4826)
```

Where 1.4826 is the consistency factor that makes MAD comparable to standard deviation
under normality. Flag observations where this score falls below a negative threshold
(e.g. -2.5 or -3.0) as drop anomalies.

**Why this works:**
- Accounts for diurnal seasonality (baseline is hour-specific)
- Robust to outliers in the baseline (median/MAD not pulled by extreme values)
- Zero hours are handled naturally: if midnight is usually zero, median=0 and a zero
  is not flagged; only a zero that is unusual for that hour gets flagged
- Simple enough to explain to a non-technical audience in 2 sentences

### Additional Detection Layer: Contextual Zero Flagging
Flag hours where:
- The value is zero (or near-zero)
- BUT the hourly median for that series/hour is meaningfully above zero
- i.e. "this hour normally has activity, but nothing happened"

This catches cases where a corrupted list produces zero hits even when transaction
volume is normal — a particularly dangerous failure mode.

### Exclusions
- **ListB_field5:** Exclude from anomaly detection entirely (essentially always zero —
  no baseline to detect drops from). Document as inactive series.
- **ListB_field6:** Flag for special treatment — high missingness, unstable values,
  values > 1. Analyse separately with appropriate caveats.
- **NaN values:** Do not flag as anomalies. Flag separately as data collection issues.

---

## Project Structure

```
swift-screening-anomaly-detection/
│
├── CLAUDE.md                  ← This file
├── README.md                  ← Project overview, setup, how to run
├── .gitignore                 ← Exclude data files (proprietary)
│
├── data/
│   └── .gitkeep              ← Data not committed (proprietary Swift data)
│
├── notebooks/
│   └── 01_exploration.ipynb  ← EDA only, for understanding
│
├── src/
│   ├── __init__.py
│   ├── load.py               ← Data loading and validation
│   ├── baseline.py           ← Hourly baseline fitting (median/MAD per series/hour)
│   ├── anomaly.py            ← Anomaly scoring and flagging logic
│   └── visualise.py          ← All plots (reusable, publication-quality)
│
├── outputs/
│   ├── figures/              ← Saved plots for PowerPoint
│   └── anomalies.csv         ← Final flagged anomalies table
│
├── main.py                   ← End-to-end pipeline runner
├── requirements.txt
└── ai_usage_log.md           ← AI tool usage documentation (required by Swift)
```

---

## Step-by-Step Workflow

### Step 1 — Data Loading & Validation (`src/load.py`)
- Load CSV with UTC datetime index
- Validate shape (480 rows × 15 columns)
- Report: missing counts per series, zero counts per series, date range
- Add `hour_of_day` and `day_of_week` columns
- Flag weekend gaps (confirm exclusion)
- Output: clean DataFrame + data quality summary

### Step 2 — Exploratory Data Analysis (`notebooks/01_exploration.ipynb`)
- Plot each series over time (time series overview)
- Plot hourly mean profile per series (show diurnal pattern)
- Heatmap: hour-of-day × series → median hit rate (shows which series/hours are active)
- Distribution plots per series (show skewness, zero inflation)
- Missing value map (which hours/series have NaN)
- Correlation heatmap (confirm independence of series)
- **Output:** 4–5 key plots for the PowerPoint EDA section

### Step 3 — Baseline Fitting (`src/baseline.py`)
- For each series × hour-of-day combination:
  - Collect all non-NaN, non-excluded values
  - Compute: median, MAD, count of observations
  - Store in a baseline DataFrame (shape: 24 × 15, two tables: median and MAD)
- Handle edge cases:
  - MAD = 0 (series is always the same value at that hour): use a small floor epsilon
  - Count < 3 observations: flag as low-confidence baseline
- **Output:** baseline_median.csv, baseline_mad.csv

### Step 4 — Anomaly Scoring (`src/anomaly.py`)
- For each observation, compute robust z-score vs its hour-of-day baseline
- Apply drop threshold (default: z < -2.5)
- Apply contextual zero flag (value ≈ 0 AND hourly median > threshold)
- Exclude: ListB_field5, NaN values
- Treat ListB_field6 separately with wider threshold
- Output per anomaly: timestamp, series, observed value, expected median, z-score, flag type
- **Output:** outputs/anomalies.csv

### Step 5 — Visualisation (`src/visualise.py`)
For each flagged series, produce:
- Time series plot with baseline band (median ± 2.5 MAD) and flagged points highlighted
- Summary table: anomaly count per series, worst drops, timestamps
- Optional: small-multiple overview of all 15 series with flags

### Step 6 — Pipeline Runner (`main.py`)
- Runs steps 1 → 5 end to end
- Saves all figures to `outputs/figures/`
- Prints summary of flagged anomalies to console
- Should run cleanly with: `python main.py`

---

## Coding Standards

- Python 3.10+
- Dependencies: pandas, numpy, matplotlib, seaborn, scipy (minimal stack)
- All functions documented with docstrings
- No hardcoded paths — use `pathlib.Path`
- Figures: consistent style (Swift-appropriate: clean, professional, not flashy)
  - Figure size: 12×5 for time series, 10×8 for heatmaps
  - Font: readable at PowerPoint slide size
  - Colour for anomalies: red markers on a grey/blue baseline band
- All randomness seeded (if any)
- `requirements.txt` pinned versions

---

## PowerPoint Sections (parallel track)

1. **Context & Problem** — What is sanctions screening, what is hit rate, why drops matter
2. **The Data** — What we have, period, structure, key data quality findings
3. **Understanding the Metric** — Hit rate formula, why it can exceed 1, hidden denominator caveat
4. **Methodology** — Why robust hourly z-score, what alternatives were considered
5. **Exploratory Analysis** — Diurnal pattern, series profiles, data quality map
6. **Results** — Flagged anomalies per series, time series plots, worst events
7. **Business Interpretation** — What a flag means operationally, recommended action
8. **Limitations & Next Steps** — Hidden denominator, no ground truth, what more data would enable

---

## Known Limitations (to address explicitly)

1. **Hidden denominator:** Hit rate is a ratio; underlying volume is unknown. Cannot
   distinguish genuine drop from low-volume noise without request count data.
2. **No ground truth:** Cannot compute precision/recall. Validation is qualitative
   (do flags make visual sense?) and synthetic (inject known drops, check recovery).
3. **Small baseline:** Only 20 observations per hour-of-day per series. Baseline
   estimates have uncertainty, especially for sparse hours.
4. **ListB_field5:** Inactive series — no anomaly detection possible.
5. **ListB_field6:** High missingness and instability — results should be treated
   with caution.
6. **Weekend exclusion:** Assumed correct per brief — weekend behaviour unknown.

---

## AI Usage Log

See `ai_usage_log.md` for full record of Claude interactions, prompts used,
outputs validated, and adjustments made — as required by Swift's case instructions.

---

*Data is proprietary Swift operational data provided under confidentiality for
this hiring exercise. Raw data files are excluded from version control.*
