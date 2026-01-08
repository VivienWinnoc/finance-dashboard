import numpy as np
import pandas as pd


def compute_returns(prices):
    # simple daily returns
    return prices.pct_change().dropna()


def _clean_weights(weights, assets):
    # weights can be dict or list/np array
    if weights is None:
        w = np.ones(len(assets)) / len(assets)
        return pd.Series(w, index=assets)

    if isinstance(weights, dict):
        w = pd.Series(weights, dtype=float)
        w = w.reindex(assets).fillna(0.0)
    else:
        w = pd.Series(list(weights), index=assets, dtype=float)

    s = float(w.sum())
    if s == 0:
        w[:] = 1.0 / len(assets)
    else:
        w = w / s

    return w


def simulate_portfolio(prices, weights=None, start_value=100.0, rebalance="none"):
    """
    rebalance:
      - "none" (buy & hold)
      - "monthly"
      - "weekly"
    """
    assets = list(prices.columns)
    w0 = _clean_weights(weights, assets)

    rets = compute_returns(prices)

    # init holdings from first day
    v = float(start_value)
    holdings = (v * w0) / prices.iloc[0]

    port = []

    for t in range(1, len(prices)):
        dt = prices.index[t]

        # update value with current prices
        v = float((holdings * prices.iloc[t]).sum())

        do_reb = False
        if rebalance == "weekly":
            do_reb = (dt.weekday() == 0)  # Monday
        elif rebalance == "monthly":
            do_reb = (dt.day == 1)

        if do_reb:
            # reset holdings to target weights at current prices
            holdings = (v * w0) / prices.iloc[t]

        port.append(v)

    port = pd.Series(port, index=prices.index[1:], name="portfolio_value")
    return port, rets
