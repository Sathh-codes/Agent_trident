"""
Alpha Vantage Stock Data Fetcher
Fetches real-time and historical stock market data.
"""

import requests
import time
import random
from datetime import datetime, timedelta

API_KEY = "UANYM86XMU4N40IV"
BASE_URL = "https://www.alphavantage.co/query"


def get_quote(symbol: str) -> dict:
    """Get the latest price snapshot for a symbol."""
    url = f"{BASE_URL}?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        quote = data.get("Global Quote", {})
        if not quote:
            return generate_mock_quote(symbol)

        price = float(quote.get("05. price", 0))
        open_idx = float(quote.get("02. open", 0))
        high = float(quote.get("03. high", 0))
        low = float(quote.get("04. low", 0))
        vol = int(quote.get("06. volume", 0))
        prev_close = float(quote.get("08. previous close", 0))
        change = float(quote.get("09. change", 0))
        change_pct = quote.get("10. change percent", "0%")

        return {
            "symbol": symbol.upper(),
            "price": price,
            "open": open_idx,
            "high": high,
            "low": low,
            "volume": vol,
            "previous_close": prev_close,
            "change": change,
            "change_percent": change_pct,
        }
    except Exception as e:
        print(f"[stock_data] get_quote error: {e}")
        return generate_mock_quote(symbol)


def get_intraday(symbol: str, interval: str = "5min") -> list:
    """Fetch intraday data using Alpha Vantage. Returns list of dicts."""
    url = f"{BASE_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={API_KEY}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        ts_key = f"Time Series ({interval})"
        if ts_key not in data:
            return generate_mock_intraday(symbol)
            
        time_series = data[ts_key]
        result = []
        for ts_str, ohlcv in sorted(time_series.items()):
            result.append({
                "time": ts_str,
                "open": float(ohlcv.get("1. open", 0)),
                "high": float(ohlcv.get("2. high", 0)),
                "low": float(ohlcv.get("3. low", 0)),
                "close": float(ohlcv.get("4. close", 0)),
                "volume": int(ohlcv.get("5. volume", 0)),
            })
        return result
    except Exception as e:
        print(f"[stock_data] get_intraday error: {e}")
        return generate_mock_intraday(symbol)


def get_daily(symbol: str) -> list:
    """Fetch daily data using Alpha Vantage."""
    url = f"{BASE_URL}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        ts_key = "Time Series (Daily)"
        if ts_key not in data:
            return get_intraday(symbol)
            
        time_series = data[ts_key]
        result = []
        for ts_str, ohlcv in sorted(time_series.items()):
            result.append({
                "time": ts_str,
                "open": float(ohlcv.get("1. open", 0)),
                "high": float(ohlcv.get("2. high", 0)),
                "low": float(ohlcv.get("3. low", 0)),
                "close": float(ohlcv.get("4. close", 0)),
                "volume": int(ohlcv.get("5. volume", 0)),
            })
        return result
    except Exception as e:
        print(f"[stock_data] get_daily error: {e}")
        return get_intraday(symbol)


def get_top_gainers_losers() -> dict:
    """Fetch trending stocks on Alpha Vantage to serve as market overview."""
    url = f"{BASE_URL}?function=TOP_GAINERS_LOSERS&apikey={API_KEY}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        
        if "top_gainers" not in data:
            return {"top_gainers": [], "top_losers": [], "most_active": []}
            
        return {
            "top_gainers": [{"ticker": item["ticker"], "name": item["ticker"], "change_percentage": item["change_percentage"]} for item in data.get("top_gainers", [])[:5]],
            "top_losers": [{"ticker": item["ticker"], "name": item["ticker"], "change_percentage": item["change_percentage"]} for item in data.get("top_losers", [])[:5]],
            "most_active": [{"ticker": item["ticker"], "name": item["ticker"], "change_percentage": item["change_percentage"]} for item in data.get("most_actively_traded", [])[:5]],
        }
    except Exception as e:
        print(f"[stock_data] trending error: {e}")
        return {"top_gainers": [], "top_losers": [], "most_active": []}


def compute_technical_indicators(prices: list) -> dict:
    """Compute basic technical indicators from price data."""
    if not prices or len(prices) < 2:
        return {}

    closes = [p["close"] for p in prices]

    # Simple Moving Averages
    sma_5 = sum(closes[-5:]) / min(5, len(closes)) if len(closes) >= 1 else 0
    sma_20 = sum(closes[-20:]) / min(20, len(closes)) if len(closes) >= 1 else 0

    # RSI (14-period)
    gains, losses = [], []
    for i in range(1, min(15, len(closes))):
        diff = closes[-i] - closes[-i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))

    avg_gain = sum(gains) / 14 if gains else 0
    avg_loss = sum(losses) / 14 if losses else 0.001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Price momentum
    current = closes[-1]
    prev = closes[-2] if len(closes) >= 2 else current
    momentum = ((current - prev) / prev * 100) if prev != 0 else 0

    # Volatility (std dev of last 20 closes)
    n = min(20, len(closes))
    mean = sum(closes[-n:]) / n
    variance = sum((c - mean) ** 2 for c in closes[-n:]) / n
    volatility = variance ** 0.5

    return {
        "sma_5": round(sma_5, 2),
        "sma_20": round(sma_20, 2),
        "rsi": round(rsi, 2),
        "momentum_pct": round(momentum, 4),
        "volatility": round(volatility, 4),
        "current_price": round(current, 2),
        "price_vs_sma20": "above" if current > sma_20 else "below",
    }


def generate_mock_intraday(symbol: str) -> list:
    """Mock random walk data for 60 periods to handle API constraints."""
    base_price = sum(ord(c) for c in symbol)  # Deterministic seed base
    history = []
    
    current_time = datetime.now() - timedelta(minutes=60*5)
    current_price = base_price * 1.5

    for i in range(60):
        # random walk
        change = random.uniform(-0.015, 0.015)
        volatility = current_price * change
        
        open_p = current_price
        high_p = current_price + abs(volatility * random.uniform(0.5, 1.5))
        low_p = current_price - abs(volatility * random.uniform(0.5, 1.5))
        close_p = current_price + volatility

        history.append({
            "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": int(random.uniform(1000, 100000))
        })
        current_price = close_p
        current_time += timedelta(minutes=5)
        
    return history


def generate_mock_quote(symbol: str) -> dict:
    """Mock stock quote."""
    history = generate_mock_intraday(symbol)
    latest = history[-1]
    prev_close = history[-2]["close"]
    change = latest["close"] - prev_close
    return {
        "symbol": symbol,
        "price": latest["close"],
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "volume": latest["volume"],
        "previous_close": prev_close,
        "change": round(change, 2),
        "change_percent": f"{(change / prev_close) * 100:.2f}%"
    }
