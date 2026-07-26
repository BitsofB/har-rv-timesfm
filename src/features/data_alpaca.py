"""
Intraday price data acquisition via Alpaca (primary data source).

Free tier gives real historical minute bars via the IEX feed (not the full
SIP consolidated tape) with generous rate limits — no funding required,
paper-trading API keys are sufficient for market-data-only access.

Requires environment variables:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import os
from datetime import datetime

import pandas as pd


def _get_client():
    """
    Lazy import + client construction so this module can be imported
    without alpaca-py installed if only other parts of the pipeline are
    being used.
    """
    from alpaca.data.historical import StockHistoricalDataClient

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables. "
            "Free paper-trading keys from https://alpaca.markets are sufficient "
            "for market data access."
        )
    return StockHistoricalDataClient(api_key, secret_key)


def fetch_intraday_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe_minutes: int = 5,
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Fetch historical intraday OHLCV bars for a single symbol.

    Returns a DataFrame indexed by timestamp with columns:
    open, high, low, close, volume, trade_count, vwap

    Note: free tier uses feed='iex' (single-venue). The full consolidated
    tape ('sip') requires a paid subscription — IEX-only data will have
    somewhat different (typically slightly wider) prices/volumes than SIP,
    worth keeping in mind when interpreting realized volatility magnitudes.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _get_client()

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(timeframe_minutes, TimeFrame.Unit.Minute),
        start=start,
        end=end,
        feed=feed,
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    # bars.df is multi-indexed (symbol, timestamp) for multi-symbol requests;
    # normalize to a single-symbol, timestamp-indexed frame
    if isinstance(df.index, pd.MultiIndex):
        df = df.loc[symbol]

    return df


def fetch_and_save(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe_minutes: int = 5,
    out_dir: str = "data/raw",
) -> str:
    """Fetch bars and save to data/raw/{symbol}_{timeframe}min.parquet."""
    df = fetch_intraday_bars(symbol, start, end, timeframe_minutes)
    path = f"{out_dir}/{symbol}_{timeframe_minutes}min.parquet"
    df.to_parquet(path)
    return path


if __name__ == "__main__":
    raise SystemExit(
        "This module is a library — call fetch_intraday_bars() or "
        "fetch_and_save() from a pipeline script, not directly."
    )
