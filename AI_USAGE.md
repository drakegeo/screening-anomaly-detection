# Note on AI use

I used Claude (an AI coding assistant) while building this, mostly as a faster way
to write boilerplate and as someone to argue with about the analysis. The modelling
decisions and the way I've read the results are mine — I checked everything against
the actual data before trusting it. A few notes on where it actually helped, since
you asked.

Most of the real value was in two places: getting plotting and refactoring done
quickly, and having something push back on my reasoning so I didn't talk myself into
a bad decision. Where it was wrong or overconfident, I overrode it — a couple of those
cases below, because they're the more interesting ones.

**Picking the baseline.** Early on I asked it to argue the case for median + MAD vs
mean + std given how skewed and zero-heavy the hit rates are. Useful for sharpening
the argument, but I only committed after plotting the distributions myself and seeing
the diurnal pattern — that's what actually sold me on a per-hour baseline, not the
model's say-so.

**The independence question.** I wanted to be sure a per-series model was the right
call and not just the easy one. I used it to think through why PCA answers that (it
compresses via correlation, so flat variance = independence), then ran it: PC1 is only
~18%, and you need 11 of 14 components for 90%. That settled it.

**Isolation Forest — where I disagreed with it.** The model initially wanted to sell
the Isolation Forest as a second detector catching things the z-score missed. I didn't
buy it: on independent features it mostly just re-derives the univariate ranking, which
is exactly why it agrees with the z-score on 11 of 44 flags. So I kept it, but framed
it honestly as corroboration plus SHAP reason codes for compliance — not as extra
detections. That framing is deliberate and I'll defend it.

**Validation framing.** I asked whether my synthetic drop-injection counts as
independent validation. It doesn't — it's a sensitivity/power analysis (how many
injected drops of each size do I recover), not label-based precision/recall. I say so
in the limitations rather than dressing it up.

**Everything else** — cleaning up the plots, splitting outputs per pipeline, removing
dead code, the .gitignore — was straightforward stuff I reviewed as it went.

I also dropped things the model suggested that added complexity without payoff: a
second PCA-based detector (PCA stays as an independence check only) and a
cross-field grid that was more noise than signal.

## A few of the actual prompts

Roughly what I typed, so you can see the split between thinking and doing:

- *"Hit rates are skewed and zero-heavy with a strong hour-of-day pattern. Argue the
  case for median + MAD over mean + std — where would each break?"*
- *"If the 15 series are independent, what's the cleanest way to prove a joint model
  is the wrong tool? Walk me through what PCA would show either way."*
- *"Be critical: does an Isolation Forest actually add detection value on independent
  features, or is it just re-deriving the univariate ranking? Don't sell it to me."*
- *"Is my synthetic drop-injection an independent validation or just a sensitivity
  analysis? Be precise about what it does and doesn't prove."*
- *"The anomaly plot has two shaded bands and it's busy — drop the redundant one and
  bump the font sizes."* (boilerplate, reviewed the diff)
- *"Split the outputs into per-pipeline folders and strip any dead code."* (mechanical)

The pattern: I used it hardest for the *reasoning* prompts — and I phrased those to
make it disagree with me, not agree. The build prompts were low-stakes and I read
every change before keeping it.

Short version: it got me from idea to working code faster and kept me honest on the
reasoning. The judgement calls — what to use, what to trust, what to cut, how to be
honest about limits — were mine.
