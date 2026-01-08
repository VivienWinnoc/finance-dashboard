import numpy as np
import pandas as pd


def corr_matrix(returns):
    # correlation matrix
    return returns.corr()


def annualized_vol(returns, periods=252):
    # annual vol (std * sqrt(252))
    return returns.std() * np.sqrt(periods)


def annualized_return(returns, periods=252):
    # approx annual return
    return returns.mean() * periods


def portfolio_returns(asset_returns, weights):
    # weights is pd.Series or list
    w = pd.Series(weights, index=asset_returns.columns, dtype=float)
    s = float(w.sum())
    if s != 0:
        w = w / s
    return (asset_returns * w).sum(axis=1)


def cum_perf(returns, start_value=100.0):
    # cumulative perf from returns
    return (1.0 + returns).cumprod() * float(start_value)


def max_drawdown(series):
    # max drawdown on a value series
    s = series.astype(float)
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min())


def simple_summary(asset_returns, portfolio_value=None):
    # quick summary table
    out = {}

    out["ann_ret"] = annualized_return(asset_returns)
    out["ann_vol"] = annualized_vol(asset_returns)

    if portfolio_value is not None:
        pv = portfolio_value.pct_change().dropna()
        out["port_ann_ret"] = annualized_return(pv)
        out["port_ann_vol"] = annualized_vol(pv)
        out["port_mdd"] = max_drawdown(portfolio_value)

    return pd.DataFrame(out)
