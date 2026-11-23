"""Global and local explanations from four methods, all on the same fitted model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import partial_dependence, permutation_importance


def global_importances(model, X_train, X_test, y_test) -> pd.DataFrame:
    sv = shap.TreeExplainer(model).shap_values(X_test)
    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=0).importances_mean
    out = pd.DataFrame({
        "impurity": model.feature_importances_,
        "permutation": perm,
        "mean |SHAP|": np.abs(sv).mean(axis=0),
    }, index=X_test.columns)
    return out / out.sum()  # shares, so the columns are comparable


def shap_local(model, X) -> np.ndarray:
    return shap.TreeExplainer(model).shap_values(X)


def lime_local(model, X_train, X, seed=0, num_samples=2000) -> np.ndarray:
    ex = LimeTabularExplainer(X_train.to_numpy(), feature_names=list(X_train.columns), mode="regression",
                              discretize_continuous=False, random_state=seed)
    out = np.zeros(X.shape)
    for i, row in enumerate(X.to_numpy()):
        exp = ex.explain_instance(row, model.predict, num_features=X.shape[1], num_samples=num_samples)
        # regression explanations live under label 1; label 0 is a placeholder LIME also fills in
        for j, w in exp.as_map()[1]:
            # LIME weights are slopes on standardized features; turn them into contributions
            out[i, j] = w * (row[j] - ex.scaler.mean_[j]) / ex.scaler.scale_[j]
    return out


def pdp_and_ice(model, X, feature: str, grid: int = 30):
    pd_res = partial_dependence(model, X, [feature], kind="both", grid_resolution=grid)
    return pd_res["grid_values"][0], pd_res["average"][0], pd_res["individual"][0]
