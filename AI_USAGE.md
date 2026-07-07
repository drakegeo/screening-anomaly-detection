# AI-Assisted Development — Notes

I used an AI coding assistant (Claude) as a pair-programmer during this exercise.
The methodology, the modelling decisions, and every interpretation of the results
are my own. I used AI to move faster on implementation and to stress-test my own
reasoning — then validated its output against the data before keeping anything.

## How I worked with it

- **I decided the approach first**, then used AI to accelerate the parts that don't
  need judgement: plotting boilerplate, refactoring, docstrings, output plumbing.
- **I treated every suggestion as a hypothesis to check**, not an answer. Where the
  model's reasoning was convincing I kept it; where it overclaimed, I pushed back
  and corrected it (examples below).
- **I validated numerically.** Any claim that made it into the analysis is backed by
  a value I confirmed in the data or a test I ran, not by the model's assertion.

## Representative interactions

**1. Choice of robust baseline**
> *"Hit rates are skewed and zero-heavy with a strong hour-of-day pattern — argue for
> the right baseline statistic."*

Used it to pressure-test my instinct for median + MAD over mean + std. **Validated**
by plotting the distributions and confirming the diurnal structure myself before
committing to a per-series, per-hour baseline.

**2. Justifying the per-series design with PCA**
> *"If the 15 series are independent, how do I prove a joint model is the wrong tool?"*

This sharpened the framing that PCA measures shared structure. **I ran it and read the
result myself**: PC1 explains only ~18%, and 11 of 14 components are needed for 90%
variance — independence, so per-series modelling is correct.

**3. Keeping the Isolation Forest honest**
> *"Does an Isolation Forest add real detection value on independent features, or does
> it just recover the univariate ranking?"*

The model initially oversold it. **I pushed back** and settled on the honest position:
on independent series it largely *confirms* the z-score (11/44 agreement), so its value
is corroboration + SHAP reason codes for compliance — not new detections. I framed the
slide that way deliberately.

**4. Validation framing**
> *"Is my synthetic drop-injection an independent validation, or a sensitivity analysis?"*

Used it to keep myself honest: it's a power analysis (what fraction of injected drops of
each size are recovered), not label-based precision/recall. I state this as a limitation
rather than overclaiming.

**5. Repo hygiene and plots**
Straightforward acceleration: cleaning up the anomaly plots, per-pipeline output folders,
removing dead code, `.gitignore`. Low-stakes, fully reviewed.

## What I adjusted or rejected

- Cut a second PCA-based detector — it added complexity without signal once independence
  was established. PCA stays as an independence proof only.
- Rejected AI phrasings that overstated the Isolation Forest and the validation; both are
  presented with their real limitations.
- Simplified visualisations the model over-engineered (redundant shading, extra panels).

**Bottom line:** AI shortened the path from idea to working code and gave me a sparring
partner for the reasoning. The judgement calls — method selection, what to trust, what to
cut, how to present limitations — are mine.
