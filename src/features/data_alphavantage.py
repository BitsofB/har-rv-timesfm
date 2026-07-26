"""
Alpha Vantage — secondary data source, used only for auxiliary/fundamental
data (economic indicators, earnings, sentiment) that isn't gated behind the
Premium tier. NOT used for the core price/RV pipeline — see
src/features/data_alpaca.py for that.

Free tier constraints: ~25 requests/day, 5 requests/minute. Batch calls
sparingly and cache results — do not call this in a tight loop.

Requires environment variable: ALPHAVANTAGE_API_KEY
"""

import os
import time

import requests

BASE_URL = "https://www.alphavantage.co/query"


def _get_api_key() -> str:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set ALPHAVANTAGE_API_KEY. Free key: "
            "https://www.alphavantage.co/support/#api-key"
        )
    return key


def _call(function: str, **params) -> dict:
    """Single API call with basic rate-limit courtesy delay."""
    query = {"function": function, "apikey": _get_api_key(), **params}
    resp = requests.get(BASE_URL, params=query, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Note" in data or "Information" in data:
        # Alpha Vantage returns HTTP 200 even when rate-limited; the message
        # is embedded in the JSON body instead of an error status
        raise RuntimeError(f"Alpha Vantage rate limit or API notice: {data}")

    time.sleep(12)  # stay comfortably under 5 calls/min on the free tier
    return data


def fetch_treasury_yield(interval: str = "monthly", maturity: str = "10year") -> dict:
    """Economic indicator — free tier accessible."""
    return _call("TREASURY_YIELD", interval=interval, maturity=maturity)


def fetch_cpi(interval: str = "monthly") -> dict:
    """CPI — free tier accessible."""
    return _call("CPI", interval=interval)


def fetch_earnings(symbol: str) -> dict:
    """Historical + estimated EPS — free tier accessible."""
    return _call("EARNINGS", symbol=symbol)


def fetch_news_sentiment(tickers: str) -> dict:
    """News & sentiment scores for given ticker(s), comma-separated."""
    return _call("NEWS_SENTIMENT", tickers=tickers)


if __name__ == "__main__":
    raise SystemExit(
        "This module is a library of auxiliary-data functions — "
        "call the individual fetch_* functions from a pipeline script."
    )
