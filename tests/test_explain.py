import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from data import make, true_contributions
from explain import global_importances, lime_local, pdp_and_ice, shap_local


@pytest.fixture(scope="module")
def fitted():
    X, y = make(n=2000)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0).fit(X[:1500], y[:1500])
    return m, X[:1500], X[1500:], y[1500:]


def test_true_contributions_sum_to_centred_target():
    X, _ = make(n=5000)
    x = X.to_numpy()
    f = 3 * x[:, 0] + 2 * x[:, 1] ** 2 + 4 * x[:, 2] * x[:, 3]
    np.testing.assert_allclose(true_contributions(X).sum(axis=1), f - 2 / 3, atol=1e-9)


def test_noise_features_get_little_importance(fitted):
    m, Xtr, Xte, yte = fitted
    g = global_importances(m, Xtr, Xte, yte)
    assert (g.loc[["x5_noise", "x6_noise", "x7_noise"]] < 0.03).all().all()


def test_shap_tracks_truth(fitted):
    m, _, Xte, _ = fitted
    rows = Xte.iloc[:100]
    sv, truth = shap_local(m, rows), true_contributions(rows)
    assert np.corrcoef(sv[:, 0], truth[:, 0])[0, 1] > 0.9


def test_lime_sign_is_right_for_linear_feature(fitted):
    m, Xtr, Xte, _ = fitted
    rows = Xte.iloc[:20]
    lv = lime_local(m, Xtr, rows, num_samples=800)
    assert np.corrcoef(lv[:, 0], rows["x0_linear"])[0, 1] > 0.9


def test_pdp_hides_interaction_that_ice_shows(fitted):
    m, _, Xte, _ = fitted
    _, avg, ind = pdp_and_ice(m, Xte, "x2_interact")
    assert (avg.max() - avg.min()) < 0.5 < ind.std(axis=0).mean()
