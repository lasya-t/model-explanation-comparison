"""Grade explanation methods against the known truth of the data-generating function."""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from data import make, true_contributions
from explain import global_importances, lime_local, pdp_and_ice, shap_local

X, y = make()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
model = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
print(f"model test R^2: {model.score(Xte, yte):.3f}\n")

print("1) Global importance (share of total)")
g = global_importances(model, Xtr, Xte, yte)
print(g.round(3).to_string())

print("\n2) Local explanations vs the true Shapley values (200 test rows)")
rows = Xte.iloc[:200]
truth = true_contributions(rows)
t0 = time.perf_counter(); sv = shap_local(model, rows); t_shap = time.perf_counter() - t0
t0 = time.perf_counter(); lv = lime_local(model, Xtr, rows, seed=0); t_lime = time.perf_counter() - t0
lv2 = lime_local(model, Xtr, rows, seed=1)
for name, est, secs in (("SHAP (tree)", sv, t_shap), ("LIME", lv, t_lime)):
    err = np.abs(est - truth).mean()
    corr = np.mean([np.corrcoef(est[:, j], truth[:, j])[0, 1] for j in range(4)])
    print(f"  {name:12s} mean abs error {err:.3f}   avg corr with truth (x0-x3) {corr:.3f}   {secs:6.2f}s")
stab = np.abs(lv - lv2).mean() / np.abs(lv).mean()
print(f"  LIME run-to-run change with a different seed: {stab:.0%} of the average contribution")
print(f"  SHAP is deterministic: run-to-run change 0%")

print("\n3) Partial dependence vs ICE")
for feat in ("x1_quadratic", "x2_interact"):
    grid, avg, ind = pdp_and_ice(model, Xte, feat)
    spread = ind.std(axis=0).mean()
    print(f"  {feat:14s} PDP range {avg.max() - avg.min():.2f}   ICE spread across rows {spread:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, feat in zip(ax, ("x1_quadratic", "x2_interact")):
    grid, avg, ind = pdp_and_ice(model, Xte, feat)
    for line in ind[:60]:
        a.plot(grid, line - line.mean(), c="gray", alpha=0.25, lw=0.7)
    a.plot(grid, avg - avg.mean(), c="crimson", lw=2.5, label="PDP (average)")
    a.set(title=feat, xlabel=feat)
    a.legend()
fig.suptitle("ICE lines (grey) vs partial dependence (red), centred")
fig.tight_layout(); fig.savefig("pdp_vs_ice.png", dpi=110)
print("\nsaved pdp_vs_ice.png")
