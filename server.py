"""
Flask Server — REST API for the AI Trading Agent System
Serves the dashboard and exposes endpoints for start/stop/status.
"""

import threading
import time
import json
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import stock_data
import agents

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ─── In-Memory State ───────────────────────────────────────────────────────────

state = {
    "running": False,
    "balance": 10000.00,  # Starting balance
    "current_symbol": None,
    "last_decision": None,
    "last_analysis": None,
    "last_update": None,
    "price_history": [],
    "trade_log": [],
    "cycle_count": 0,
    "error": None,
    "agent_logs": [],  # Visible output statements from agents
    "portfolio": {},   # Track currently owned stocks
}

analysis_thread = None
stop_event = threading.Event()

# ─── Stock Selection ────────────────────────────────────────────────────────────

DEFAULT_WATCHLIST = ["bitcoin", "ethereum", "solana", "ripple", "cardano", "dogecoin", "polkadot"]
_used_stocks = set()  # Track stocks already analyzed this session


def select_stock() -> str:
    """Pick a NEW stock to analyze each cycle. Rotates through top gainers & watchlist."""
    global _used_stocks
    try:
        market = stock_data.get_top_gainers_losers()
        # Try top gainers first
        for entry in market.get("top_gainers", [])[:10]:
            ticker = entry.get("ticker", "")
            if ticker and ticker not in _used_stocks:
                _used_stocks.add(ticker)
                return ticker
        # Try most active
        for entry in market.get("most_active", [])[:10]:
            ticker = entry.get("ticker", "")
            if ticker and ticker not in _used_stocks:
                _used_stocks.add(ticker)
                return ticker
    except Exception:
        pass
    # Fallback to watchlist rotation
    for sym in DEFAULT_WATCHLIST:
        if sym not in _used_stocks:
            _used_stocks.add(sym)
            return sym
    # All exhausted — reset and start over
    _used_stocks.clear()
    _used_stocks.add(DEFAULT_WATCHLIST[0])
    return DEFAULT_WATCHLIST[0]


def add_agent_log(agent_name: str, message: str):
    """Add a visible output statement from an agent."""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "agent": agent_name,
        "message": message,
    }
    state["agent_logs"].append(entry)
    # Keep last 100 log entries
    state["agent_logs"] = state["agent_logs"][-100:]


# ─── Background Analysis Loop ──────────────────────────────────────────────────

def analysis_loop():
    """Runs in a background thread. Every 2 minutes: pick NEW stock → fetch data → agents analyze."""
    global state

    while not stop_event.is_set():
        try:
            state["error"] = None

            # 1. Select a NEW stock each cycle
            symbol = select_stock()
            state["current_symbol"] = symbol
            add_agent_log("System", f"🔍 Scanning market... Selected new stock: {symbol}")
            print(f"[server] Cycle starting — selected stock: {symbol}")

            # 2. Fetch stock data
            add_agent_log("System", f"📊 Fetching real-time data for {symbol}...")
            quote = stock_data.get_quote(symbol)
            intraday = stock_data.get_intraday(symbol, "5min")
            market_overview = stock_data.get_top_gainers_losers()

            if not quote:
                state["error"] = f"Could not fetch data for {symbol}. Market may be closed."
                add_agent_log("System", f"⚠️ No quote data for {symbol} — market may be closed.")
                if not intraday:
                    add_agent_log("System", "⏳ No data available. Waiting for next cycle...")
                    stop_event.wait(300)
                    continue

            # 3. Compute indicators
            indicators = stock_data.compute_technical_indicators(intraday)

            # Store price history for chart
            state["price_history"] = intraday[-60:]

            # 4. Run multi-agent analysis with logging
            add_agent_log("Technical Analyst", f"🔬 Analyzing {symbol} charts, RSI, moving averages...")
            add_agent_log("Sentiment Analyst", f"🧠 Reading market mood and sector trends for {symbol}...")
            add_agent_log("Risk Manager", f"🛡️ Evaluating risk/reward profile for {symbol}...")

            analysis = agents.run_full_analysis(
                symbol=symbol,
                stock_data=quote or {"symbol": symbol, "price": intraday[-1]["close"] if intraday else 0},
                indicators=indicators,
                market_overview=market_overview,
                balance=state["balance"],
            )

            # 5. Log each agent's output
            agents_data = analysis.get("agents", {})
            for key, agent_data in agents_data.items():
                name = agent_data.get("agent", key.title())
                action = agent_data.get("action", "HOLD")
                conf = agent_data.get("confidence", 0)
                reasoning = agent_data.get("reasoning", "")
                add_agent_log(name, f"📋 My verdict: {action} (confidence: {conf}%) — {reasoning[:150]}")

            # 6. Update state
            final = analysis.get("final_decision", {})
            state["last_decision"] = {
                "action": final.get("final_action", "HOLD"),
                "confidence": final.get("final_confidence", 0),
                "summary": final.get("summary", ""),
                "dissenting_views": final.get("dissenting_views", ""),
            }
            state["last_analysis"] = agents_data
            state["last_update"] = datetime.now().isoformat()
            state["cycle_count"] += 1

            # Log final decision
            final_action = final.get("action", "HOLD")
            final_conf = final.get("confidence", 0)
            add_agent_log("Risk Manager", f"⚖️ Final consensus: {final_action} on {symbol} (confidence: {final_conf}%)")
            if final.get('summary'):
                add_agent_log("Risk Manager", f"💬 {final.get('summary')}")

            # 7. Execute trade and update portfolio
            confidence = final.get("confidence", 50)
            if quote:
                price = quote.get("price", 0)
                if price > 0:
                    if final_action == "BUY":
                        # Spend 10% of current balance based on confidence
                        spend_amount = state["balance"] * 0.10 * (confidence / 100)
                        shares_to_buy = spend_amount / price
                        if shares_to_buy > 0:
                            state["balance"] -= spend_amount
                            if symbol not in state["portfolio"]:
                                state["portfolio"][symbol] = {"shares": 0, "avg_price": 0}
                            
                            curr_shares = state["portfolio"][symbol]["shares"]
                            curr_avg = state["portfolio"][symbol]["avg_price"]
                            new_shares = curr_shares + shares_to_buy
                            new_avg = ((curr_shares * curr_avg) + spend_amount) / new_shares
                            
                            state["portfolio"][symbol]["shares"] = new_shares
                            state["portfolio"][symbol]["avg_price"] = new_avg
                            
                            add_agent_log("System", f"💰 BOUGHT {shares_to_buy:.4f} shares of {symbol} at ${price:.2f}. Balance: ${state['balance']:,.2f}")
                    elif final_action == "SELL":
                        if symbol in state["portfolio"] and state["portfolio"][symbol]["shares"] > 0:
                            shares_to_sell = state["portfolio"][symbol]["shares"]
                            revenue = shares_to_sell * price
                            state["balance"] += revenue
                            del state["portfolio"][symbol]
                            add_agent_log("System", f"💰 SOLD all {shares_to_sell:.4f} shares of {symbol} at ${price:.2f}. Balance: ${state['balance']:,.2f}")
                        else:
                            add_agent_log("System", f"⏸️ SELL signal, but no shares of {symbol} owned.")
                    else:
                        add_agent_log("System", f"⏸️ HOLD — no position change. Balance: ${state['balance']:,.2f}")

                state["trade_log"].append({
                    "time": datetime.now().isoformat(),
                    "symbol": symbol,
                    "action": final_action,
                    "price": price,
                    "confidence": confidence,
                    "balance_after": round(state["balance"], 2),
                })
                state["trade_log"] = state["trade_log"][-50:]

            add_agent_log("System", f"✅ Cycle {state['cycle_count']} complete. Waiting 5 min for next scan...")
            print(f"[server] Cycle {state['cycle_count']} complete. "
                  f"Decision: {final_action} on {symbol} | Balance: ${state['balance']:,.2f}")

        except Exception as e:
            state["error"] = str(e)
            add_agent_log("System", f"❌ Error during analysis: {str(e)}")
            print(f"[server] Analysis error: {e}")

        # Wait 5 minutes before next cycle
        stop_event.wait(300)


# ─── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/api/status")
def get_status():
    """Return the current system state."""
    return jsonify({
        "running": state["running"],
        "balance": round(state["balance"], 2),
        "symbol": state["current_symbol"],
        "decision": state["last_decision"],
        "agents": state["last_analysis"],
        "last_update": state["last_update"],
        "cycle_count": state["cycle_count"],
        "error": state["error"],
        "agent_logs": state["agent_logs"][-30:],  # Last 30 log entries
        "portfolio": state["portfolio"],
    })


@app.route("/api/start", methods=["POST"])
def start_analysis():
    """Start the background analysis loop."""
    global analysis_thread
    if state["running"]:
        return jsonify({"status": "already_running"}), 200

    stop_event.clear()
    state["running"] = True
    state["error"] = None
    analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
    analysis_thread.start()
    return jsonify({"status": "started"}), 200


@app.route("/api/stop", methods=["POST"])
def stop_analysis():
    """Stop the background analysis loop."""
    global analysis_thread
    if not state["running"]:
        return jsonify({"status": "already_stopped"}), 200

    stop_event.set()
    state["running"] = False
    if analysis_thread:
        analysis_thread.join(timeout=5)
        analysis_thread = None
    return jsonify({"status": "stopped"}), 200


@app.route("/api/history")
def get_history():
    """Return price history for the chart."""
    return jsonify({
        "symbol": state["current_symbol"],
        "prices": state["price_history"],
        "trades": state["trade_log"],
    })


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Trading Agent System")
    print("  Dashboard: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
