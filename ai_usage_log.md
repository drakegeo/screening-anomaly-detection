# AI Usage Log
**Project:** Screening Hit Rate Anomaly Detection — Swift Take-Home Case  
**Candidate:** George Drakoulas  
**AI Tool:** Claude (Anthropic) via Claude Code  

---

## Purpose

This log documents all AI assistance received during this case, as required by Swift's submission instructions. For each interaction, it records what was prompted, what the AI produced, and what was validated or modified by the candidate.

---

## Session Summary

### Phase 1 — Architecture and Method Design

**Prompt intent:** Reviewed the CLAUDE.md specification and discussed methodology options. Asked the AI to explain the robust hourly z-score approach and why it was preferred over alternatives (moving average, ARIMA, isolation forest).

**AI output:**
- Explained the per-series, per-hour-of-day baseline design
- Justified median + MAD over mean + std for skewed/zero-heavy hit rate distributions
- Recommended the 1.4826 MAD consistency factor
- Proposed the three-tier flag taxonomy (drop_zscore / contextual_zero / contextual_zero_holiday)
- Identified ListB_field5 (inactive) and ListB_field6 (problematic) for special handling

**Candidate validation:**
- Agreed with median/MAD approach — matches standard practice for financial time series
- Verified the flag taxonomy covers all relevant business scenarios
- Confirmed holiday tagging to Dec 25 and Jan 1 based on inspection of the data

---

### Phase 2 — Project Structure

**Prompt intent:** Defined the codebase structure: two `src/` subfolders (`processing/` and `detection/`), two pipeline entry points, `pyproject.toml` instead of `requirements.txt`.

**AI output:**
- Generated `src/processing/load.py`, `src/processing/eda.py`
- Generated `src/detection/baseline.py`, `src/detection/anomaly.py`, `src/detection/visualise.py`
- Generated `pipelines/processing.py`, `pipelines/detection.py`
- Generated `pyproject.toml` with `hatchling` build backend

**Candidate validation:**
- Reviewed all module boundaries — processing vs detection separation was the candidate's decision
- Verified `load.py` constants (INACTIVE_SERIES, PROBLEMATIC_SERIES, KNOWN_HOLIDAYS)
- Confirmed `pyproject.toml` dependencies match installed environment

---

### Phase 3 — EDA Visualisations

**Prompt intent:** Generated 6 EDA plots. Iterated on layout, colour scheme, and legend placement.

**AI output:** Plots 01–06:
1. Time series small multiples (all 15 series)
2. Diurnal heatmap (raw + normalised)
3. Data quality map (missing/zero/present)
4. Hourly profiles (median ± IQR per hour)
5. Value distributions (log-scale boxplots)
6. Spearman correlation heatmap

**Bug fixed by AI:** `plt.cm.get_cmap` removed in matplotlib 3.11 — fixed to `matplotlib.colormaps["RdYlGn_r"].resampled(3)`.

**Candidate validation:**
- Reviewed all 6 plots for correctness
- Confirmed normalisation logic in plot 02 (per-series to [0,1] for comparability)
- Confirmed Spearman (not Pearson) for non-normal hit rate distributions

---

### Phase 4 — Baseline Fitting and Anomaly Scoring

**Prompt intent:** Implemented `baseline.py` and `anomaly.py`. Iterated on threshold sensitivity.

**AI output:**
- `fit_baseline()`: median + scaled MAD per series × hour, with MAD floor (1e-6) and low-confidence flag (< 3 obs)
- `score_anomalies()`: z-score scoring, contextual zero logic, holiday awareness
- `threshold_sensitivity()`: comparison at z = −2.0 / −2.5 / −3.0

**Candidate decisions:**
- Chose −2.5 as default threshold after reviewing sensitivity plot — balances precision vs recall
- Confirmed CONTEXTUAL_ZERO_MIN_MEDIAN = 0.005 as floor to avoid noise from near-zero medians
- Decided ListB_field6 gets wider threshold (−2.0) and is excluded from contextual zero

---

### Phase 5 — Anomaly Visualisations (extensive iteration)

**Prompt intent:** Multiple rounds of iteration on `visualise.py` to achieve publication quality.

**Design decisions driven by candidate:**
- Red fill = alarm zone (below threshold), grey fill = expected range — candidate rejected earlier all-blue designs
- Hollow `○` for critical anomaly flags (colour = hour of day), black `×` for holiday zeros
- Bottom panel shows 24h diurnal profile with coloured vertical lines at critical alarm hours only (holiday zeros excluded after candidate identified the confusion)
- Fixed two-entry legend on every plot for consistency
- Figure width extended to 22in to reduce marker overlap
- Hour labels removed from top panel markers (colour linkage to bottom panel sufficient)
- Negative y lower bound (−3% of max) so zero-value markers sit above x-axis spine

**Bugs caught and fixed by candidate:**
- Misleading legend colours (flag-type colours on markers that were actually hour-coloured)
- Holiday zeros generating bottom-panel vertical lines when no critical flag present — candidate identified this was confusing; fixed by separating critical_hours from holiday_hours
- `×` markers at y=0 invisible (clipped by x-axis spine) — fixed with small negative ylim

---

### Phase 6 — Pipeline Runners and Documentation

**Prompt intent:** Final pipeline scripts and this documentation.

**AI output:** `pipelines/processing.py`, `pipelines/detection.py`, `README.md`, `ai_usage_log.md`

**Candidate validation:**
- Ran both pipelines end-to-end and verified outputs
- All figures reviewed manually before committing

---

## Assessment of AI Contribution

| Component | AI wrote | Candidate directed / validated |
|-----------|----------|-------------------------------|
| Method choice (median/MAD) | Proposed and explained | Agreed — matches known best practice |
| Code structure | Generated | Architecture decisions were candidate's |
| Bug fixes | Fixed when identified | Candidate identified all bugs |
| Plot design | Iterated on prompts | All visual design decisions were candidate's |
| Business interpretation | Explained on request | Candidate applied domain knowledge |
| Flag taxonomy | Proposed | Candidate approved and refined |

AI acted as a fast code generator and debugging partner. All analytical decisions, design choices, and validation were performed by the candidate.
