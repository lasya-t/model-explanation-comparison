# SHAP vs LIME vs Permutation vs PDP: Graded Against the Truth

Model-explanation methods are usually compared by eye. Here the data comes from a **known formula**, so each method's answer can be checked against the exact truth.

```
y = 3·x0 + 2·x1² + 4·x2·x3 + noise
x4 = near-copy of x0 (correlated, but no effect of its own)
x5, x6, x7 = pure noise
```

The model is gradient boosting (400 trees, depth 4) with test R² = 0.938. Each feature is designed to test something different: a linear effect, a non-linear one, an interaction, a correlated decoy, and noise.

## 1. Global importance (share of total)

| Feature | Impurity | Permutation | mean \|SHAP\| |
|---|---|---|---|
| x0 linear | 0.571 | 0.469 | 0.483 |
| x1 quadratic | 0.085 | 0.060 | **0.170** |
| x2 interaction | 0.132 | 0.237 | 0.128 |
| x3 interaction | 0.152 | 0.228 | 0.129 |
| x4 copy of x0 | **0.048** | **0.006** | **0.064** |
| x5–x7 noise | ≤ 0.005 | ≈ 0 | ≤ 0.010 |

- All three methods push the noise features to near zero.
- **The correlated copy (x4) is where they differ.** Impurity and SHAP give it some credit, because the trees sometimes split on it instead of x0. Permutation importance gives it almost none: shuffling x4 hurts nothing, since x0 carries the same information. Neither answer is wrong; they answer different questions ("what did the model use?" vs "what would hurt if it were lost?").

## 2. Local explanations vs the exact Shapley values (200 rows)

| Method | Mean abs error | Correlation with truth | Time | Same answer on re-run? |
|---|---|---|---|---|
| **SHAP (TreeExplainer)** | **0.114** | **0.978** | 0.06 s | yes, deterministic |
| LIME | 0.194 | 0.972 | 2.7 s | **no**: 11% change with a different seed |

TreeSHAP is more accurate, about 45× faster on trees, and stable. LIME fits a new local linear model from random samples each time, so its explanation changes between runs. That's a real problem if a reason code shown to a customer or regulator changes when you ask again.

**A bug worth knowing about:** the first version read LIME's regression weights from `as_map()[0]`, and LIME's correlation with the truth came out at **−0.97**. In regression mode LIME also fills in a placeholder label `0`; the real weights are under label `1`. The fix is its own commit, and a test now checks the sign.

## 3. Partial dependence hides interactions

| Feature | PDP range | ICE spread across rows |
|---|---|---|
| x1 (quadratic) | 1.62 | 1.97 |
| **x2 (interaction)** | **0.16** | **2.02** |

The PDP for x2 is **almost flat**, which suggests x2 doesn't matter. In fact it matters a lot, but its effect points up or down depending on the sign of x3, and averaging cancels the two out. ICE curves (one line per row) show the hidden fan shape (`pdp_vs_ice.png`). **Always plot ICE alongside a PDP.**

## How to run

```bash
pip install -r requirements.txt
python compare.py     # ~30 seconds
pytest
```

## Next steps
- Add SHAP interaction values to split the x2·x3 credit explicitly
- Add accumulated local effects (ALE) plots, which handle correlated features better than PDP
