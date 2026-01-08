import pandas as pd
import requests
from io import StringIO


def load_price_data(tickers, start_date=None):
    # Load daily close prices for a list of assets
    # data from Stooq (public, no api key)

    prices = []

    for ticker in tickers:
        url = f"https://stooq.pl/q/d/l/?s={ticker}&i=d"
        r = requests.get(url)

        if r.status_code != 200:
            raise ValueError("data download failed for " + ticker)

        df = pd.read_csv(StringIO(r.text))
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        df = df[["Close"]]
        df.columns = [ticker.upper()]

        prices.append(df)

    prices = pd.concat(prices, axis=1).dropna()

    if start_date is not None:
        prices = prices[prices.index >= start_date]

    return prices

