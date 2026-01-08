import numpy as np
import pandas as pd


def _infer_step(index: pd.DatetimeIndex):
    """Infère un pas de temps (Timedelta) à partir de l'index."""
    if len(index) < 2:
        return pd.Timedelta(days=1)

    freq = pd.infer_freq(index)
    if freq is not None:
        try:
            return pd.tseries.frequencies.to_offset(freq).delta
        except Exception:
            pass

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return pd.Timedelta(days=1)
    return deltas.median()


def linear_price_prediction(prices: pd.Series, horizon: int = 30, lookback: int = 60) -> pd.Series:
    """
    Prédiction linéaire simple (sans sklearn).
    - prices: série de prix (DatetimeIndex)
    - horizon: nb de points à prédire
    - lookback: nb de points récents utilisés pour ajuster la droite
    """
    if prices is None or len(pd.Series(prices).dropna()) < 5:
        raise ValueError("Pas assez de données pour prédire.")

    p = pd.Series(prices).dropna()
    if not isinstance(p.index, pd.DatetimeIndex):
        raise ValueError("L'index de prices doit être un DatetimeIndex.")

    p = p.iloc[-min(int(lookback), len(p)) :]

    t0 = p.index[0]
    x = (p.index - t0).total_seconds().astype(float)
    y = p.values.astype(float)

    a, b = np.polyfit(x, y, deg=1)

    step = _infer_step(p.index)
    last_t = p.index[-1]
    future_index = pd.date_range(start=last_t + step, periods=int(horizon), freq=step)

    xf = (future_index - t0).total_seconds().astype(float)
    yf = a * xf + b

    return pd.Series(yf, index=future_index, name="prediction")
