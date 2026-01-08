import numpy as np
import pandas as pd


def linear_price_prediction(
    prices: pd.Series,
    horizon: int = 30
) -> pd.Series:
    """
    Prédiction linéaire simple (régression via numpy).

    prices  : série de prix historiques
    horizon : nombre de périodes futures à prédire
    """

    prices = prices.dropna()

    if len(prices) < 10:
        raise ValueError("Pas assez de données pour effectuer une prédiction")

    # Temps (0, 1, 2, ...)
    x = np.arange(len(prices))
    y = prices.values

    # Régression linéaire y = a*x + b
    a, b = np.polyfit(x, y, 1)

    # Projection future
    x_future = np.arange(len(prices), len(prices) + horizon)
    y_future = a * x_future + b

    future_index = pd.date_range(
        start=prices.index[-1],
        periods=horizon + 1,
        freq=prices.index.inferred_freq or "D"
    )[1:]

    return pd.Series(y_future, index=future_index, name="Prediction")
