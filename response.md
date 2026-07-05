# Plan & Presentation Guide — Screening Hit-Rate Anomaly Detection

This document is the narrative spine for the PowerPoint and the implementation plan
for the second (working) multivariate method. It covers all four pieces in order:
**(1) processing/EDA, (2) primary method — robust hourly z-score, (3) PCA validation,
(4) the second method that works — Isolation Forest + SHAP.**

---

## 0. TL;DR — the story in one line

> Normal depends on the *hour of day*, and the 15 series are *independent*. So we
> detect drops **per series against an hour-of-day baseline** (robust z-score). We
> then prove independence two ways (correlation + PCA), which is *why* a linear
> multivariate model fails — and *why* a **tree-based** multivariate model
> (Isolation Forest), which thrives on independent features, is the right second
> lens. SHAP turns each machine flag into a plain-language reason code for compliance.

---

## 1. Honest assessment of the proposed pipeline

The proposal is a generic AML/transaction-monitoring template. Half of it does not
fit our data; half is genuinely valuable. Keep the valuable half.

| Proposed idea | Fits our data? | Decision |
|---------------|----------------|----------|
| Per-entity / per-customer scaling | ✗ no customers/entities exist | Drop |
| 7/30/90-day rolling windows | ✗ only 20 weekdays of data | Drop |
| Outbound-wire / SAR feedback loop | ✗ no transactions, no labels | Drop (note as future work) |
| Velocity / rate-of-change features | ~ partial — hit-rate deltas exist | Optional feature |
| **Isolation Forest** | ✓ **designed for independent features** | **Adopt** |
| **SHAP explainability** | ✓ compliance needs reason codes | **Adopt** |
| Extreme Value Theory thresholding | ~ sound but heavy for 480 points | Simplify → robust quantile |
| Operational-capacity triage (top-N) | ✓ realistic for ops teams | Adopt as framing |

**Why Isolation Forest is the correct second method (not PCA):**
PCA reduces dimensionality by exploiting *correlation*. Our series are independent
(PC1 = 18%, needs 11/14 components), so PCA cannot compress and adds nothing.
Isolation Forest is the opposite — it isolates points by randomly splitting on
*single* features. Independent, sharply-separable features get isolated near the
root of the trees, so independence is a *strength*, not a weakness. This is the
clean contrast that makes the methodology story compelling.

---

## 2. Presentation structure (slide-by-slide)

### Section A — Problem & Data  *(uses processing pipeline + EDA plots)*
1. **Business context** — sanctions screening; a silent hit-rate *drop* = a list may
   be corrupted and transactions sail through unscreened. Drops matter more than spikes.
2. **The data** — 480 hours × 15 series (5 lists × 3 fields), weekdays only,
   2023-12-18 → 2024-01-12. Hit rate can exceed 1; zero = no requests; NaN = collection failure.
3. **Key data-quality findings** *(plot 03 data-quality map)* — ListB_field5 inactive,
   ListB_field6 30% missing, sparse zero-heavy fields.
4. **The hidden-denominator caveat** — we only see the ratio, not request volume. Hour-of-day
   proxies volume (business hours = reliable, overnight = noisy).
5. **Diurnal pattern** *(plot 02 heatmap, plot 04 hourly profiles)* — "normal" is
   hour-specific. This motivates an hour-of-day baseline.

### Section B — Method 1: Robust Hourly Z-Score  *(primary detector)*
6. **The idea** — for each series and each hour-of-day, fit median + MAD from the 20
   weekday samples. Score `z = (obs − median)/(MAD × 1.4826)`. Flag `z < −2.5`.
7. **Why robust** — hit-rate distributions are skewed and zero-heavy; median/MAD are
   not dragged by outliers or the Dec-25 spike.
8. **Three flag types** — drop_zscore, contextual_zero (active hour, no hits),
   contextual_zero_holiday (Dec 25 / Jan 1 = volume effect, de-prioritised).
9. **Threshold defence** *(plot 07 sensitivity)* — flag counts at −2.0/−2.5/−3.0;
   −2.5 balances noise vs recall.
10. **Reading a result** *(plot anomaly_lista_field3)* — two panels: time series with
    alarm zone + 24h diurnal profile; marker colour links a flag to its hour.

### Section C — Why not a linear multivariate model? (PCA validation)
11. **The question** — could we model the 15 series jointly instead of one-by-one?
12. **Two independent answers: no.** Correlation heatmap *(plot 06)* ≈ 0; PCA scree
    *(plot 09)* PC1 = 18%, needs 11/14 comps for 90%. Series are independent.
13. **Consequence** — linear reduction (PCA) cannot help (agrees with z-score on 1/44).
    This *justifies* the per-series univariate design. **Turn a "failed" method into a
    proof that the architecture is right.**

### Section D — Method 2: Isolation Forest + SHAP  *(non-linear multivariate that works)*
14. **The pivot** — independence breaks PCA but *suits* tree isolation. Isolation Forest
    on the standardised-residual matrix flags hours whose joint pattern is extreme.
15. **What it adds over z-score** — a single multivariate score per hour + automatic
    ranking; catches combinations the per-series test rates as individually mild.
16. **SHAP reason codes** — for each flagged hour, SHAP attributes the score to
    specific series ("ListA_field3 contributed +0.4 of the anomaly score"). This is the
    explanation a compliance officer needs; because features are uncorrelated, SHAP is
    clean (no multicollinearity ambiguity).
17. **Agreement with z-score** — report overlap; consensus = high confidence, machine-only
    flags = new candidates, z-only = point drops the ensemble smoothed over.

### Section E — Results & corroboration
18. **Corroboration grid** *(plot 08)* — for no list do fields drop *together* on the same
    date → **no list-level failure**; every real anomaly is field-scoped.
19. **The two real drops** — ListA_field3 (sustained from Jan 2, recurs at 10:00) and
    ListD_field11 (Dec 29, Jan 12). Holiday flags = volume.

### Section F — Business interpretation, recommendation, limitations
20. **Alerting rule** — escalate only when ≥2 fields of a list drop together; log
    single-field drops. Under this rule no false alarm fires here — the correct outcome.
21. **Confidence** — corroborated across statistically distinct methods (robust z-score
    + Isolation Forest), and validated by the PCA independence proof. Not "proved" (no
    ground truth) but strongly triangulated.
22. **Limitations & next steps** — hidden denominator (add request counts = #1 upgrade),
    no labels (build the analyst-feedback loop → semi-supervised over time), short history.

---

## 3. Implementation plan — Method 2 (Isolation Forest + SHAP)

### 3.1 Design
- **Input:** reuse the standardised-residual matrix already built in `pca.py`
  (`(obs − hourly_median)/hourly_MAD`, NaN→0, ListB_field5 excluded). This keeps the
  ensemble consistent with the diurnal baseline — it scores *deviation from normal*,
  not the raw diurnal pattern.
- **Model:** `sklearn.ensemble.IsolationForest(n_estimators=300, contamination='auto',
  random_state=42)`. Fit on the full residual matrix; output `score_samples` (higher =
  more normal) → invert to an anomaly score per hour.
- **Direction filter:** we only care about *drops*. For each flagged hour, keep it only
  if the dominant contributing series has a **negative** residual (a fall, not a spike).
- **Threshold:** robust quantile on the anomaly score (e.g. top 5%, or median + 3·MAD),
  plus an "operational top-N" view (sort scores, take the N the team can review).
- **Explainability:** `shap.TreeExplainer` on the Isolation Forest → per-hour SHAP
  values → the series with the largest positive contribution is the reason code.

### 3.2 New files / functions
```
src/detection/isoforest.py
    build_residual_matrix()          # (reuse from pca.py or import)
    fit_isolation_forest(...)        # -> per-hour anomaly score DataFrame
    shap_reason_codes(model, X)      # -> per-hour top-contributing series + value
    consensus_with_zscore(...)       # overlap with primary detector

src/detection/visualise.py
    plot_isoforest_scores(...)       # score timeline + threshold + z-score overlay
    plot_shap_summary(...)           # global SHAP bar (which series drive anomalies)
    plot_shap_reasons(...)           # per top-N flag: SHAP reason-code bars

pipelines/detection.py
    wire in fit -> score -> shap -> plots, write outputs/isoforest_scores.csv
```

### 3.3 New dependencies (tradeoff to confirm)
- `scikit-learn` (IsolationForest) — reasonable, widely trusted.
- `shap` — heavier install; pulls in numba. Justified by the compliance
  explainability requirement. **Decision needed:** accept `shap`, or approximate reason
  codes with per-hour SHAP-free contribution (residual magnitude per series) to keep the
  stack minimal. Recommended: add `shap` — the reason-code slide is worth it.

### 3.4 Outcome — BUILT ✓
Implemented in `src/detection/isoforest.py`, wired into the pipeline, plots 10–12.
Results confirm the prediction:
- **15 drop-hours flagged**; **11 of 44** z-score critical flags independently caught
  (vs **1/44** for PCA) — a genuine, statistically distinct second opinion.
- SHAP global importance is roughly even across all series (0.13–0.40, no dominant
  driver) → reinforces independence.
- Top alerts align with the known events (Jan 1 ListA_field1 z=−6.56, Jan 5 10:00
  ListA_field3, Jan 3 02:00 ListC_field9).
- Core conclusion unchanged: **no list-level failure**; ListA_field3 and ListD_field11
  remain the real field-level drops. The ensemble *strengthens* confidence, not overturns.

**Plots produced:**
- `10_isoforest_scores.png` — anomaly-score timeline, flags, z-score agreement
- `11_shap_importance.png` — global SHAP: which series drive the detector
- `12_shap_reasons.png` — per-alert SHAP reason-code heatmap (compliance slide)

---

## 4. Decisions taken
1. **`shap` added** ✓ (numba 0.66 supports Python 3.14; numpy pinned to 2.4.6).
2. **Threshold** — robust median + 3·MAD on the anomaly score. Operational top-N view
   available via `TOP_N` in `isoforest.py` for the reason-code slide.
3. **Velocity feature** — skipped (marginal value; keeps the model simple/explainable).

## 5. Final plot inventory for the deck
| # | File | Slide use |
|---|------|-----------|
| 01–06 | EDA plots | Section A — data & diurnal pattern |
| 07 | threshold sensitivity | Section B — threshold defence |
| anomaly_*.png | per-series z-score | Section B — reading a result |
| 08 | corroboration grid | Section E — list vs field level |
| 09 | PCA diagnostic | Section C — independence proof |
| 10 | Isolation Forest scores | Section D — multivariate that works |
| 11 | SHAP global importance | Section D — which series drive it |
| 12 | SHAP reason codes | Section D — per-alert explanation (compliance) |
