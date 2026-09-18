# Mood Congruency Causal Effect Summary

- Source: `v3/results/raw/mood_congruency/Qwen_Qwen2.5-1.5B-Instruct_mood_congruency_ambiguous.jsonl`
- Total observations: 3210

## Linear Slope $\beta_{\mathrm{mood}}$ (Change in Recognized Valence per $\sigma$ Steering)

| Layer | Direction | N stimuli | $\beta_{\mathrm{mood}}$ (Slope) | SE | $t$-statistic | $p$-value | $\Delta V$ ($-3\sigma$) | $\Delta V$ ($+3\sigma$) |
|---|---|---|---|---|---|---|---|---|
| 14 | `valence` | 107 | **-0.0323** | 0.0016 | -20.77 | 1.18e-70 | +0.084 | -0.109 |
| 14 | `random` | 107 | **-0.0042** | 0.0013 | -3.32 | 9.63e-04 | +0.017 | -0.008 |
| 16 | `valence` | 107 | **-0.0169** | 0.0012 | -14.40 | 6.20e-40 | +0.045 | -0.057 |
| 16 | `random` | 107 | **+0.0444** | 0.0013 | 34.50 | 6.40e-138 | -0.132 | +0.134 |
| 20 | `valence` | 107 | **+0.0244** | 0.0013 | 18.87 | 3.23e-61 | -0.080 | +0.067 |
| 20 | `random` | 107 | **+0.0091** | 0.0009 | 10.50 | 1.45e-23 | -0.028 | +0.027 |


### Interpretation Guide
- **Mood Congruency Confirmed**: $\beta_{\mathrm{mood}} > 0$ with $p < 0.01$ in `valence` condition, while `random` condition $\approx 0$.
- **Negative finding**: $\beta_{\mathrm{mood}} \approx 0$ or non-significant, indicating decoupling between internal mood state and recognition circuitry.
