import numpy as np

from src.eval.diebold_mariano import diebold_mariano_test
from src.models.har_rv import qlike, qlike_loss


def test_equal_accuracy_gives_high_pvalue():
    rng = np.random.default_rng(1)
    # same mean, independent noise -> no systematic difference in accuracy
    loss_a = rng.normal(0.2, 0.05, size=200)
    loss_b = rng.normal(0.2, 0.05, size=200)
    stat, p_value = diebold_mariano_test(loss_a, loss_b)
    assert p_value > 0.05


def test_systematically_worse_model_gives_significant_pvalue():
    rng = np.random.default_rng(0)
    loss_a = rng.normal(0.1, 0.01, size=200)   # consistently lower loss
    loss_b = rng.normal(0.5, 0.01, size=200)   # consistently higher loss
    stat, p_value = diebold_mariano_test(loss_a, loss_b)

    assert stat < 0  # negative -> model A (lower loss) favored
    assert p_value < 0.01


def _reference_dm(loss_a, loss_b, h=1):
    """
    Independent, unvectorized reimplementation of the same formula used as
    a ground truth to pin exact numeric values -- deliberately written as
    plain Python loops (not numpy broadcasting) so a ddof/off-by-one/mean
    bug in the implementation under test isn't invisible to a test that
    happens to share its arithmetic.
    """
    d = [a - b for a, b in zip(loss_a, loss_b)]
    n = len(d)
    d_bar = sum(d) / n

    gamma0 = sum((x - d_bar) ** 2 for x in d) / n
    var_d = gamma0
    for lag in range(1, h):
        terms = [(d[t] - d_bar) * (d[t - lag] - d_bar) for t in range(lag, n)]
        gamma_k = sum(terms) / n
        var_d += 2 * (1 - lag / h) * gamma_k

    var_d_bar = var_d / n
    dm_stat = d_bar / var_d_bar ** 0.5

    hln_correction = ((n + 1 - 2 * h + h * (h - 1) / n) / n) ** 0.5
    dm_stat *= hln_correction

    from scipy import stats
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return dm_stat, p_value


def test_matches_independent_reference_implementation_h1():
    rng = np.random.default_rng(7)
    loss_a = rng.normal(0.3, 0.1, size=40)
    loss_b = rng.normal(0.25, 0.12, size=40)

    stat, p_value = diebold_mariano_test(loss_a, loss_b, h=1)
    ref_stat, ref_p = _reference_dm(loss_a, loss_b, h=1)

    assert np.isclose(stat, ref_stat)
    assert np.isclose(p_value, ref_p)


def test_matches_independent_reference_implementation_h3():
    rng = np.random.default_rng(11)
    loss_a = rng.normal(0.3, 0.1, size=40)
    loss_b = rng.normal(0.25, 0.12, size=40)

    stat, p_value = diebold_mariano_test(loss_a, loss_b, h=3)
    ref_stat, ref_p = _reference_dm(loss_a, loss_b, h=3)

    assert np.isclose(stat, ref_stat)
    assert np.isclose(p_value, ref_p)


def test_qlike_loss_mean_matches_qlike():
    y_true = np.array([1.0, 2.0, 3.0, 0.5])
    y_pred = np.array([1.1, 1.8, 3.5, 0.4])
    assert np.isclose(qlike_loss(y_true, y_pred).mean(), qlike(y_true, y_pred))


def test_qlike_loss_perfect_prediction_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert np.allclose(qlike_loss(y, y), 0.0)
