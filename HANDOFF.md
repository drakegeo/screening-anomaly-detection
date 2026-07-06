# Session Handoff — Screening Anomaly Detection

Context carrier for resuming on another machine. Committed to the repo so it travels via git.
Read this + `STORYTELLING.md` + `CLAUDE.md` to reload full context.

---

## Project state: COMPLETE and defensible

Take-home for **Swift Senior Applied Data Scientist**. Detect **drop anomalies** in
sanctions screening hit-rate data (15 series = 5 lists × 3 fields, 480 hours, 20 weekdays,
2023-12-18 → 2024-01-12 UTC). Remaining work is **rehearsal, not code.**

---

## How to run (on the new machine)

```bash
python -m venv .venv
.venv/Scripts/activate           # Windows;  source .venv/bin/activate on mac/linux
pip install -e .                 # pulls scikit-learn, shap, etc. from pyproject.toml

python -m pipelines.processing   # 6 EDA plots (01-06)
python -m pipelines.detection    # z-score, PCA, IsoForest+SHAP, corroboration, events, dashboard
python -m pipelines.validation   # synthetic injection recovery curve (plot 13)
```

Data file `data/screening_hitrate.csv` is **not committed** (proprietary). Copy it manually.
Run with `.venv/Scripts/python.exe -m ...` if sklearn/shap not found on base interpreter.

---

## The methodology arc (the story)

1. **Independence proof** (correlation + PCA) → justifies per-series univariate design.
   PC1 = 18%, needs 11/14 components for 90%. Series are independent.
2. **Robust Hourly Z-Score** (primary): `z = (obs − hourly_median) / (MAD × 1.4826)`, flag `z < −2.5`.
   Per series, per hour-of-day. Median/MAD because distributions skewed + zero-heavy.
3. **Synthetic validation**: analytical drop injection `obs = median × (1−f)`. Recovery curve.
   0.25% FP rate; precision-tuned. Weak recall on subtle drops = hidden-denominator limitation.
4. **Isolation Forest + SHAP** (2nd detector): thrives on independence (PCA fails on it).
   Drops-only encoding (`X = min(X_full, 0)`). 11/44 z-score agreement vs 1/44 for PCA.
5. **Corroboration grid**: no list drops all fields together → **no list-level failure**.
6. **Dashboard** (plot 14): health grid + KPI + event-ranked alert queue = system, not just model.

---

## Key results (event-level, the honest count)

- **57 hour-flags → 42 events (17 non-holiday).** Count events, not hours.
- **No list-level failure** — every anomaly field-scoped.
- Top offenders (ranked by event view):
  - **ListC_field9** — worst z=−4.30, 5 events, solid baseline (median 0.09). THE top alert.
  - **ListA_field3** — recurring 10:00 drops Jan 2/3/4/5. Clearest *systematic* signature.
  - **ListD_field11** — Dec 29, Jan 12.
- **All 12 top alerts on full 20-obs baselines; 0 low-confidence active cells** → no artifacts.
- Escalation rule: ≥2 fields of a list drop same date ⇒ escalate. Single field ⇒ log. Zero false escalations here.

---

## Defensive checks (interviewer-proofing)

- **Day-of-week**: adds only 11.5% residual reduction but cuts baseline 20→4 obs/cell. Hour-only justified.
- **Scaling**: z-score IS per-series scaling. PCA uses correlation-PCA (unit variance). IsoForest uses standardised-residual matrix. Global scaling would be *wrong* (don't compare ListA to ListB).
- **Histograms**: plot 05 (log-scale boxplots) already shows skew/zero-inflation. Histograms redundant.

---

## Senior talking points (defend, don't hide)

- **Cost asymmetry / threshold philosophy**: missed corruption (regulatory breach) >> false alarm (analyst time). Our −2.5 is precision-tuned; arguably should run MORE sensitive + triage. Naming the trade-off is the point.
- **Boiling-frog**: rolling baseline adapts to slow degradation and hides it. Fix: freeze golden baseline from certified-healthy period.
- **Hidden denominator**: highest-ROI upgrade is not more ML — it's logging request counts (one column).
- **Why no deep learning / forecasting**: considered, rejected. 20 days thin; 1h detection latency fine; forecasting doesn't change the action. Restraint = signal.

---

## File map

```
src/processing/load.py        data load, validation, holiday tagging
src/processing/eda.py         6 EDA plots; shared STYLE/_save/_list_color helpers
src/detection/baseline.py     fit_baseline() → median/mad/count/low_conf (24 × n_series)
src/detection/anomaly.py      score_anomalies(), group_into_events(), threshold_sensitivity()
src/detection/pca.py          _standardised_residual_matrix(), pca_reconstruction_flags()
src/detection/isoforest.py    fit_isolation_forest(), shap_reason_codes()
src/detection/validation.py   run_injection_test(), recovery_rates(), false_positive_rate()
src/detection/diagnostics.py  day_of_week_check(), alert_confidence_audit()
src/detection/visualise.py    all anomaly/diagnostic/dashboard plots (07-14 + anomaly_*)
pipelines/{processing,detection,validation}.py   entry points
```

Outputs: `outputs/figures/*.png`, `outputs/anomalies.csv`, `outputs/events.csv`,
`outputs/validation/`, `outputs/alert_confidence_audit.csv`.

---

## Docs

- `STORYTELLING.md` — 18-slide presentation guide, all plots embedded, anticipated-questions table.
- `README.md` — methodology, results, validation, limitations.
- `ai_usage_log.md` — Swift-required AI log, Phases 1-11.
- `CLAUDE.md` — original project spec.

---

## Special series (do not forget)

- `ListB_field5` — INACTIVE, excluded entirely (always zero, no baseline).
- `ListB_field6` — PROBLEMATIC, wider threshold (−2.0), excluded from contextual zero (30% missing, unstable).
- Holidays hardcoded: Dec 25 2023, Jan 1 2024.

---

## Open / optional (nothing blocking)

- Not committed yet — commit when ready.
- Could add one before/after histogram to visually motivate median/MAD (marginal, optional).
- Environment: numpy pinned 2.4.6 (shap constraint). Python 3.14.
