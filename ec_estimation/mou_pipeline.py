"""Fit mOU models across tau; select tau separately in the notebook."""
from pathlib import Path
import sys
import numpy as np
from scipy import signal
VENDOR = Path(__file__).resolve().parent / "pyMOU-master"
sys.path.insert(0, str(VENDOR))
from pymou.mou_model import MOU

def prepare_timeseries(data, tr=0.72, bandpass=(0.01, 0.1),
                       n_regions=None, mapping=None):
    """Time x regions; filter, select regions, then average grouped signals."""
    X = np.asarray(data, dtype=float)
    if X.ndim != 2 or X.shape[0] < 4 or X.shape[1] < 2:
        raise ValueError("Input must be a time x regions matrix.")
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values.")
    if tr <= 0:
        raise ValueError("TR must be positive.")
    if bandpass is not None:
        low, high = bandpass
        if not 0 < low < high < 0.5 / tr:
            raise ValueError("Bandpass must lie between zero and Nyquist.")
        b, a = signal.butter(2, [low, high], btype="bandpass", fs=1 / tr)
        X = signal.filtfilt(b, a, X, axis=0)
    if n_regions is not None:
        if not 2 <= n_regions <= X.shape[1]:
            raise ValueError("Invalid number of selected regions.")
        X = X[:, :n_regions]
    if mapping is not None:
        mapping = np.asarray(mapping).reshape(-1)
        labels = np.unique(mapping)
        if (len(mapping) != X.shape[1] or not np.isfinite(mapping).all()
                or not np.array_equal(labels, np.arange(1, len(labels) + 1))):
            raise ValueError("Mapping must contain consecutive integer labels starting at 1.")
        X = np.column_stack([X[:, mapping == k].mean(axis=1) for k in labels])
    return X


def empirical_moments(X, lag=1):
    """Reproduce calc_Corr0/calc_Corrlag in my_module.py exactly.

    Q0 uses X[:-1] even when lag differs from 1. Each segment is centered
    separately and standardized with population standard deviations (ddof=0),
    while the product is divided by segment length minus one.
    Thus Q0's diagonal is (T-1)/(T-2), not exactly one.
    Q1[i,j] relates earlier region i to later region j.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Expected finite time x regions data.")
    if not isinstance(lag, (int, np.integer)) or not 1 <= lag < X.shape[0] - 1:
        raise ValueError("lag must be a positive integer smaller than T-1.")
    def standardized(A):
        sd = A.std(axis=0)
        if np.any(sd == 0):
            raise ValueError("Constant time series encountered.")
        return (A - A.mean(axis=0)) / sd
    Z0 = standardized(X[:-1])
    Zpast = standardized(X[:-lag])
    Zfuture = standardized(X[lag:])
    Q0 = Z0.T @ Z0 / (len(Z0) - 1)
    Q1 = Zpast.T @ Zfuture / (len(Zpast) - 1)
    return Q0, Q1



def select_tau(taus, Js, Q1, iterations, stable, max_iter):
    """First continuous eligible interval; omit initial/trivial points."""
    taus = np.asarray(taus, float)
    if np.any(np.diff(taus) <= 0):
        raise ValueError("Tau grid must be strictly increasing.")
    mask = ~np.eye(len(Q1), dtype=bool)
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if (np.isfinite(a).all()
            and np.isfinite(b).all() and a.std() > 0 and b.std() > 0) else np.nan
    scores = np.array([corr(J[mask], Q1[mask]) for J in Js])
    asym_scores = np.array([corr(((J-J.T)/2)[mask], ((Q1-Q1.T)/2)[mask]) for J in Js])
    eligible = (np.asarray(stable, bool) & (iterations > 0) & (iterations < max_iter)
                & np.isfinite(scores) & np.array([np.any(J[mask] != 0) for J in Js]))
    search = np.zeros(len(taus), dtype=bool)
    excluded = np.zeros_like(search)
    ix = np.flatnonzero(eligible)
    if len(ix):
        start = int(ix[0]); end = start
        while end+1 < len(taus) and eligible[end+1]:
            end += 1
        search[start:end+1] = True
        excluded[start] = True
        # Same explicitly specified exclusion as the exploratory notebook.
        excluded |= search & (iterations <= 12) & np.isclose(asym_scores, 1, atol=1e-10, rtol=0)
        search &= ~excluded
    candidates = np.flatnonzero(search)
    best = int(candidates[np.argmax(scores[candidates])]) if len(candidates) >= 3 else -1
    boundary = best >= 0 and best in (candidates[0], candidates[-1])
    return dict(corr_J_Q1=scores, eligible=eligible, search_mask=search,
                excluded_initial=excluded, best_index=np.asarray(best),
                boundary_maximum=np.asarray(boundary),
                tau_opt=np.asarray(taus[best] if best >= 0 else np.nan))


def fit_tau_grid(data, taus, tr=0.72, bandpass=(0.01, 0.1), n_regions=100,
                mapping=None, max_iter=100, eta_C=0.005):
    """Fit independently at each tau; do not refit the selected estimate."""
    X = prepare_timeseries(data, tr, bandpass, n_regions, mapping)
    Q0, Q1 = empirical_moments(X, lag=1)
    taus = np.asarray(taus, float)
    if taus.ndim != 1 or len(taus) < 4 or not np.isfinite(taus).all() or np.any(taus <= 0) or np.any(np.diff(taus) <= 0):
        raise ValueError("Use at least four strictly increasing positive finite taus.")
    if max_iter <= 12:
        raise ValueError("max_iter must exceed 12.")
    Js = np.full((len(taus), len(Q0), len(Q0)), np.nan)
    Sigmas = np.full_like(Js, np.nan)
    iterations = np.zeros(len(taus), int)
    stable = np.zeros(len(taus), bool)
    distances = np.full(len(taus), np.nan)
    errors = [''] * len(taus)
    for i, tau in enumerate(taus):
        try:
            model = MOU()
            model.fit_LO(Q0, Q1, lag=1, prior=0.0, tau=float(tau),
                         eta_C=eta_C, max_iter=max_iter)
            J, Sigma = np.asarray(model.J), np.asarray(model.Sigma)
            if Sigma.ndim == 1:
                Sigma = np.diag(Sigma)
            iterations[i] = int(model.d_fit['iterations'])
            distances[i] = float(model.d_fit['distance'])
            Js[i], Sigmas[i] = J, Sigma
            if np.isfinite(J).all() and np.isfinite(Sigma).all() and np.isfinite(distances[i]):
                stable[i] = np.linalg.eigvals(J).real.max() < 0
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, UnboundLocalError, OverflowError) as exc:
            errors[i] = type(exc).__name__ + ': ' + str(exc)
    # Return the fitted grid without choosing tau or computing asymmetries.
    return dict(taus=taus, J_grid=Js, Sigma_grid=Sigmas, iterations=iterations,
                stable=stable, distance=distances, errors=np.asarray(errors), Q0=Q0, Q1=Q1,
                tr=np.asarray(tr), max_iter=np.asarray(max_iter))


def estimate_q1(*args, **kwargs):
    """Legacy Q1 selection, retained for older notebooks only."""
    fitted = fit_tau_grid(*args, **kwargs)
    selected = select_tau(fitted['taus'], fitted['J_grid'], fitted['Q1'],
                          fitted['iterations'], fitted['stable'], int(fitted['max_iter']))
    best = int(selected['best_index'])
    empty = np.full_like(fitted['Q0'], np.nan)
    fitted.update(selected)
    fitted.update(J_raw=fitted['J_grid'][best] if best >= 0 else empty,
                  J_target_source=fitted['J_grid'][best].T if best >= 0 else empty,
                  Sigma=fitted['Sigma_grid'][best] if best >= 0 else empty)
    return fitted



def plot_result(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    tau, r = result['taus'], result['corr_J_Q1']
    axes[0].plot(tau, r, '.-', color='0.7', label='All finite fits')
    valid = result['search_mask']
    axes[0].scatter(tau[valid], r[valid], label='Search interval')
    best = int(result['best_index'])
    if best >= 0:
        axes[0].scatter(tau[best], r[best], marker='*', s=180, color='green', label='Selected tau')
        J = result['J_raw'].copy()
        np.fill_diagonal(J, 0)
        limit = np.abs(J).max()
        im = axes[1].imshow(J, cmap='RdBu_r', vmin=-limit, vmax=limit)
        fig.colorbar(im, ax=axes[1], label='EC weight (per sample)')
    axes[0].set(xscale='log', xlabel='Tau (samples)', ylabel='Pearson r: J raw versus Q1')
    axes[0].legend(fontsize=8)
    axes[1].set(title='Selected EC (diagonal hidden)', xlabel='Target', ylabel='Source')
    return fig
