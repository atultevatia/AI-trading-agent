import os
import operator
import json
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
from news_engine import news_engine

# Load environment variables
load_dotenv()

# --- Configuration ---
CAPITAL = 10000

# --- Constants & Prompts ---
SECTOR_URLS = {
    "AUTO": "https://archives.nseindia.com/content/indices/ind_niftyautolist.csv",
    "IT": "https://archives.nseindia.com/content/indices/ind_niftyitlist.csv",
    "BANK": "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",
    "PHARMA": "https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv",
    "FMCG": "https://archives.nseindia.com/content/indices/ind_niftyfmcglist.csv"
}

FALLBACK_MAPPING = {
    "AI": ["TATAELXSI.NS", "PERSISTENT.NS", "OFSS.NS", "CYIENT.NS", "HAPPSTMNDS.NS", "KPITTECH.NS", "ZENTEC.NS"],
    "AUTO": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "ASHOKLEY.NS"],
    "BANK": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "TORNTPHARM.NS"],
    "FMCG": ["HUL.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "VBL.NS"]
}

ANALYST_AGENT_PROMPT = """You are a Lead Quantitative Research Analyst. 
Your task is to synthesize technical data and news to identify high-probability swing trades.
Move beyond simple rules; reason about the convergence of price action and sentiment.

Output JSON:
{
    "thesis": "detailed reasoning",
    "conviction": 0-100,
    "entry": float,
    "target": float,
    "stop_loss": float
}
"""

RISK_MANAGER_PROMPT = """You are a Conservative Risk Officer. Your job is to find reasons NOT to take the trade pitched by the Analyst.
Critique the entry, stop loss, and news sentiment.

Output JSON:
{
    "status": "APPROVED" | "REJECTED",
    "risk_criticism": "reasoning for selection",
    "adjusted_stop": float
}
"""

PORTFOLIO_MANAGER_PROMPT = """You are a Fund Manager. Total budget: ₹10,000.
Allocate capital across approved trades based on conviction and risk. Max 40% per stock.

Output JSON:
{
    "allocations": [
        {"ticker": "string", "shares": int, "weight_pct": float, "reason": "why"}
    ],
    "remaining_cash": float
}
"""

# --- State Definitions ---

class StockAnalysis(TypedDict):
    ticker: str
    price: float
    thesis: str
    conviction: int
    entry: float
    target: float
    stop_loss: float
    risk_status: str
    risk_criticism: str
    adjusted_stop: float

class OverallState(TypedDict):
    sector: str
    tickers: List[str]
    analyses: List[StockAnalysis]
    portfolio: List[Dict[str, Any]]
    remaining_cash: float

# --- Helper Functions ---

def fetch_nse_constituents(sector_key: str):
    url = SECTOR_URLS.get(sector_key)
    if not url: return FALLBACK_MAPPING.get(sector_key, [])
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return [f"{sym}.NS" for sym in df['Symbol'].tolist()]
        return FALLBACK_MAPPING.get(sector_key, [])
    except: return FALLBACK_MAPPING.get(sector_key, [])

# --- Nodes ---

def sector_loader_node(state: OverallState):
    sector = state['sector'].upper()
    fetch_key = "IT" if sector == "AI" else sector
    tickers = fetch_nse_constituents(fetch_key)
    return {"tickers": tickers, "analyses": [], "portfolio": [], "remaining_cash": CAPITAL}

def analyst_node(state: OverallState):
    analyses = []
    tickers = state['tickers']
    if not tickers: return {"analyses": []}

    print(f"--- Analyst: Processing {len(tickers)} stocks ---")
    data = yf.download(tickers, period="7mo", interval="1d", progress=False, group_by='ticker')
    llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})

    for ticker in tickers:
        try:
            df = data[ticker].copy().dropna() if len(tickers) > 1 else data.copy().dropna()
            if len(df) < 50: continue

            # Technicals
            close_ser = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            sma50 = close_ser.rolling(50).mean().iloc[-1]
            sma200 = close_ser.rolling(200).mean().iloc[-1]
            
            # News
            news = news_engine.get_stock_news(ticker)
            news_txt = "\n".join([n['title'] for n in news[:3]])

            msg = f"Ticker: {ticker}\nPrice: {close_ser.iloc[-1]}\nSMA50: {sma50}\nSMA200: {sma200}\nNews: {news_txt}"
            response = llm.invoke([SystemMessage(content=ANALYST_AGENT_PROMPT), HumanMessage(content=msg)])
            res = json.loads(response.content)

            analyses.append({
                "ticker": ticker,
                "price": float(close_ser.iloc[-1]),
                "thesis": res['thesis'],
                "conviction": res['conviction'],
                "entry": res['entry'],
                "target": res['target'],
                "stop_loss": res['stop_loss']
            })
        except Exception as e:
            print(f"Analyst Error on {ticker}: {e}")
            
    return {"analyses": analyses}

def risk_manager_node(state: OverallState):
    print("--- Risk Manager: Boarding Review ---")
    vetted = []
    llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    
    for analysis in state['analyses']:
        try:
            msg = f"Trade Pitch for {analysis['ticker']}:\n{json.dumps(analysis, indent=2)}"
            response = llm.invoke([SystemMessage(content=RISK_MANAGER_PROMPT), HumanMessage(content=msg)])
            res = json.loads(response.content)

            if res['status'] == "APPROVED":
                analysis.update({
                    "risk_status": "APPROVED",
                    "risk_criticism": res['risk_criticism'],
                    "adjusted_stop": res['adjusted_stop']
                })
                vetted.append(analysis)
            else:
                print(f"Risk Manager REJECTED {analysis['ticker']}: {res['risk_criticism']}")
        except Exception as e:
            print(f"Risk Manager Error: {e}")

    return {"analyses": vetted}

def portfolio_manager_node(state: OverallState):
    print("--- Portfolio Manager: Allocating Capital ---")
    if not state['analyses']: return {"portfolio": [], "remaining_cash": CAPITAL}

    llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    msg = f"Approved Trades:\n{json.dumps(state['analyses'], indent=2)}\nMax Capital: ₹{CAPITAL}"
    
    try:
        response = llm.invoke([SystemMessage(content=PORTFOLIO_MANAGER_PROMPT), HumanMessage(content=msg)])
        res = json.loads(response.content)
        return {"portfolio": res['allocations'], "remaining_cash": res.get('remaining_cash', 0)}
    except Exception as e:
        print(f"Portfolio Manager Error: {e}")
        return {"portfolio": []}

# --- Graph ---

def create_agent_graph():
    workflow = StateGraph(OverallState)
    workflow.add_node("Loader", sector_loader_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("RiskManager", risk_manager_node)
    workflow.add_node("PortfolioManager", portfolio_manager_node)

    workflow.set_entry_point("Loader")
    workflow.add_edge("Loader", "Analyst")
    workflow.add_edge("Analyst", "RiskManager")
    workflow.add_edge("RiskManager", "PortfolioManager")
    workflow.add_edge("PortfolioManager", END)

    return workflow.compile()

app = create_agent_graph()

