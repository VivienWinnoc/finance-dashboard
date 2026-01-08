import streamlit as st

from src.portfolio.data import load_price_data
from src.portfolio.portfolio import simulate_portfolio
from src.portfolio.metrics import corr_matrix, annualized_vol


def render_portfolio_page():
    st.title("Quant B — Portfolio analysis")
    st.caption("Equal-weight · configurable rebalancing · price-return only")

    st.markdown(
        """
This section implements a simple **multi-asset portfolio backtest**, 
based on **equal-weight allocations** and periodic rebalancing.

The goal is *not* to predict markets or trade live, but to compare 
how different configurations behave over time under the same assumptions.

Prices are daily closes from **Stooq (public data)**.
Dividends are **not included**, so results should be interpreted 
as price-return only.

This tool is mainly designed for **exploration and comparison**, 
not for production trading.
"""
    )

    st.sidebar.header("Quant B — Parameters")

    tickers = st.sidebar.text_input(
        "Tickers (Stooq format)",
        value="aapl.us, msft.us, googl.us",
        help="US stocks on Stooq usually end with .us",
    )

    rebalance = st.sidebar.selectbox(
        "Rebalancing frequency",
        options=["none", "monthly", "weekly"],
        index=1,
    )

    start_value = st.sidebar.number_input(
        "Initial portfolio value",
        min_value=10.0,
        value=100.0,
        step=10.0,
    )

    st.sidebar.caption(
        "Data source: Stooq daily close prices. "
        "Total returns are not available (dividends not included)."
    )

    run = st.sidebar.button("Run simulation")

    if not run:
        st.info("Choose parameters in the sidebar, then click **Run simulation**.")
        return

    tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    prices = load_price_data(tickers)

    if prices.shape[0] < 50:
        st.warning(
            "Limited amount of historical data for the selected tickers. "
            "Results may be less reliable."
        )

    port, rets = simulate_portfolio(
        prices,
        start_value=start_value,
        rebalance=rebalance,
    )

    col1, col2, col3 = st.columns(3)

    total_ret = (port.iloc[-1] / port.iloc[0]) - 1.0
    col1.metric("Total return", f"{total_ret*100:.2f}%")

    p_rets = port.pct_change().dropna()
    p_vol = float(p_rets.std() * (252 ** 0.5))
    col2.metric("Ann. vol (port)", f"{p_vol*100:.2f}%")

    col3.metric("Last value", f"{port.iloc[-1]:.2f}")

    st.caption(
        "High total returns are mainly driven by long-term compounding, "
        "equal-weight rebalancing effects, and the strong historical performance "
        "of US equities over the selected period. "
        "This does not imply similar future results."
    )

    st.subheader("Prices")
    st.line_chart(prices)

    st.subheader("Portfolio value")
    st.line_chart(port)

    st.subheader("Correlation matrix")
    st.dataframe(corr_matrix(rets))

    st.subheader("Annualized volatility (assets)")
    st.dataframe(annualized_vol(rets))
