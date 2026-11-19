"""Censored-Gaussian (Tobit) objective for LightGBM on log10 LC50.

cens codes: 0 exact, 1 right-censored (true > y, e.g. '>' limit tests), 2 left-censored (true < y).
"""
import numpy as np, lightgbm as lgb
from scipy.special import log_ndtr

LOG_SQRT2PI = 0.5 * np.log(2 * np.pi)


def _logphi(z):
    return -0.5 * z * z - LOG_SQRT2PI


def tobit_objective(cens, sigma):
    cens = np.asarray(cens)

    def fobj(preds, data):
        y = data.get_label()
        z = (y - preds) / sigma
        g = np.empty_like(preds); h = np.empty_like(preds)
        e, r, l = cens == 0, cens == 1, cens == 2
        g[e] = (preds[e] - y[e]) / sigma ** 2; h[e] = 1 / sigma ** 2
        lam = np.exp(_logphi(z[r]) - log_ndtr(-z[r]))           # phi / (1 - Phi)
        g[r] = -lam / sigma; h[r] = np.clip(lam * (lam - z[r]), 1e-6, 1) / sigma ** 2
        kap = np.exp(_logphi(z[l]) - log_ndtr(z[l]))            # phi / Phi
        g[l] = kap / sigma; h[l] = np.clip(kap * (kap + z[l]), 1e-6, 1) / sigma ** 2
        return g, h
    return fobj


def censored_loglik(y, mu, cens, sigma):
    z = (np.asarray(y) - np.asarray(mu)) / sigma
    cens = np.asarray(cens)
    ll = np.where(cens == 0, _logphi(z) - np.log(sigma), np.where(cens == 1, log_ndtr(-z), log_ndtr(z)))
    return ll


def fit_tobit_lgb(X, y, cens, params, rounds, cat, sigma=0.7):
    p = dict(params); p["objective"] = tobit_objective(cens, sigma)
    init = np.full(len(y), np.mean(np.asarray(y)[np.asarray(cens) == 0]))
    ds = lgb.Dataset(X, y, categorical_feature=cat, init_score=init)
    m = lgb.train(p, ds, rounds)
    return m, init[0]
