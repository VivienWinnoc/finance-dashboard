import pandas as pd
import requests
from io import StringIO


def load_price_data(tickers, start_date=None):
    # daily close prices from stooq (no api key)
    prices = []

    for ticker in tickers:
        url = f"https://stooq.pl/q/d/l/?s={ticker}&i=d"
        r = requests.get(url, timeout=20)

        if r.status_code != 200:
            raise ValueError("data download failed for " + ticker)

        text = r.text.strip()

        # stooq sometimes returns a text like "No data" or a html-ish page
        if ("Date" not in text) and ("Data" not in text):
            raise ValueError("unexpected response for " + ticker + " (no Date column)")

        df = pd.read_csv(StringIO(text))

        # handle different column naming
        if "Date" in df.columns:
            date_col = "Date"
        elif "Data" in df.columns:
            date_col = "Data"
        else:
            raise ValueError("bad csv format for " + ticker)

                # stooq can return different names for close price
        close_col = None
        for c in ["Close", "Zamkniecie", "Zamknięcie"]:
            if c in df.columns:
                close_col = c
                break

        if close_col is None:
            raise ValueError("missing close column for " + ticker)

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)

        df = df[[close_col]]
        df.columns = [ticker.upper()]


        prices.append(df)

    prices = pd.concat(prices, axis=1).dropna()

    if start_date is not None:
        prices = prices[prices.index >= start_date]

    return prices
