import json
import os
import time
from datetime import datetime

TRADES_FILE = "paper_trades.json"

def load_trades():
    if not os.path.exists(TRADES_FILE):
        return {"active": [], "closed": []}
    try:
        with open(TRADES_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"active": [], "closed": []}

def save_trades(trades):
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=4)

def book_trade(ticker, price, quantity, stop_loss, target, thesis):
    trades = load_trades()
    
    # Check if already active
    for t in trades['active']:
        if t['ticker'] == ticker:
            return False, f"Trade already active for {ticker}"
            
    new_trade = {
        "id": int(time.time()),
        "ticker": ticker,
        "entry_price": float(price),
        "quantity": int(quantity),
        "stop_loss": float(stop_loss),
        "target": float(target),
        "thesis": thesis,
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ACTIVE"
    }
    
    trades['active'].append(new_trade)
    save_trades(trades)
    return True, f"Booked trade for {ticker} at ₹{price}"

def close_trade(trade_id, exit_price):
    trades = load_trades()
    for i, t in enumerate(trades['active']):
        if t['id'] == trade_id:
            trade = trades['active'].pop(i)
            trade['exit_price'] = float(exit_price)
            trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['quantity']
            trade['pnl_pct'] = ((trade['exit_price'] / trade['entry_price']) - 1) * 100
            trade['status'] = "CLOSED"
            trades['closed'].append(trade)
            save_trades(trades)
            return True, "Trade closed successfully"
    return False, "Trade not found"
