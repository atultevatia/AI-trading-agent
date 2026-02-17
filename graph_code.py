import os
from typing import TypedDict, List, Annotated
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# === 1. System Prompts ===
TA_PROMPT = """You are a CMT (Chartered Market Technician) certified analyst with 15 years of experience in Indian markets. You specialize in swing trading setups (2-8 weeks).
Input: Historical price data and current price from the Market Data Fetcher.
Task: Analyze the following:
1. Trend: Identify the primary trend (Uptrend, Downtrend, Sideways) on Daily and Weekly timeframes.
2. Indicators: Calculate and interpret RSI (is it >70 or <30?), MACD (crossovers?), and Moving Averages (50DMA, 200DMA).
3. Support/Resistance: Identification of key levels.
4. Volume Analysis: Is price moving on high volume?
Output: A concise technical report (markdown) covering the above points, concluding with a Technical Bias (Bullish/Bearish/Neutral) and a Strength Score (1-10)."""

FA_PROMPT = """You are a fundamental equity research analyst focused on Indian 'New Age' Tech and Automobile sectors. You look for growth triggers and valuation comfort.
Input: Company news, financial ratios, and sector trends.
Task:
1. Catalyst Identification: Spot specific triggers (e.g., 'EV sales numbers', 'AI deal wins', 'Government PLI schemes').
2. Valuation Check: Is the stock trading at a fair P/E relative to its growth?
3. Risks: Identify any red flags (e.g., governance issues, declining margins).
4. Sector Tailwind: Does the current macro environment support this sector (e.g., interest rate cuts, budget allocation)?
Output: A fundamental note (markdown) highlighting triggers and risks. Conclude with a Fundamental Turnaround probability (High/Medium/Low)."""

RISK_PROMPT = """You are the Risk Management desk. Your job is to protect capital. You are pessimistic by nature and look for what could go wrong.
Input: Technical and Fundamental analysis reports.
Task:
1. Stop Loss Calculation: Based on technical support levels, suggest a logical stop loss.
2. Position Sizing: Recommend a max allocation (e.g., 'Do not exceed 3% of portfolio').
3. Reward-to-Risk: Calculate the potential upside vs. the downside risk. Reject trades with RR < 1:2.
4. Volatility Check: Is the stock too volatile for a swing trade?
Output: A risk assessment memo (markdown) with calculated SL levels and a Trade Approval (Approved/Rejected)."""

STRATEGY_PROMPT = """You are the Portfolio Manager making the final decision. You receive reports from your Technical Analyst, Fundamental Analyst, and Risk Manager.
Input: All prior reports.
Task:
1. Synthesize the conflicting or confirming signals.
2. Decide on the final trade: BUY, SELL, or WAIT.
3. If BUY/SELL:
    - Entry Zone: Specific price range.
    - Targets: T1 (Conservative) and T2 (Aggressive).
    - Stop Loss: As per Risk Manager.
    - Thesis: 1-sentence summary of why we are taking this trade.
Output: The final trading signal in a clear, structured markdown format."""

# === 2. State Definition ===
class AgentState(TypedDict):
    ticker: str
    sector: str
    market_data: dict
    technical_analysis: str
    fundamental_analysis: str
    risk_assessment: str
    final_recommendation: str
    messages: Annotated[List[BaseMessage], operator.add]

# === 3. Mock Tools / API Placeholders ===
def fetch_market_data(ticker: str):
    # In a real app, use yfinance or similar:
    # import yfinance as yf
    # tick = yf.Ticker(ticker)
    # return tick.info
    return {
        "price": 100.0, 
        "volume": "Above Average", 
        "news": [
            f"{ticker} announces new AI partnership.",
            "Sector outlook remains positive."
        ]
    }

# === 4. Node Functions ===
def market_data_node(state: AgentState):
    ticker = state['ticker']
    print(f"--- Fetching Market Data for {ticker} ---")
    data = fetch_market_data(ticker)
    return {"market_data": data}

def technical_analyst_node(state: AgentState):
    print("--- Technical Analyst Working ---")
    market_data = state['market_data']
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = [
        SystemMessage(content=TA_PROMPT),
        HumanMessage(content=f"Market Market Data for {state['ticker']}: {str(market_data)}")
    ]
    response = llm.invoke(messages)
    return {"technical_analysis": response.content}

def fundamental_analyst_node(state: AgentState):
    print("--- Fundamental Analyst Working ---")
    market_data = state['market_data']
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = [
        SystemMessage(content=FA_PROMPT),
        HumanMessage(content=f"Market Market Data for {state['ticker']}: {str(market_data)}")
    ]
    response = llm.invoke(messages)
    return {"fundamental_analysis": response.content}

def risk_manager_node(state: AgentState):
    print("--- Risk Manager Working ---")
    ta_data = state['technical_analysis']
    fa_data = state['fundamental_analysis']
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = [
        SystemMessage(content=RISK_PROMPT),
        HumanMessage(content=f"Technical Analysis: {ta_data}\n\nFundamental Analysis: {fa_data}")
    ]
    response = llm.invoke(messages)
    return {"risk_assessment": response.content}

def strategy_generator_node(state: AgentState):
    print("--- Portfolio Manager Finalizing ---")
    ta_data = state['technical_analysis']
    fa_data = state['fundamental_analysis']
    risk_data = state['risk_assessment']
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = [
        SystemMessage(content=STRATEGY_PROMPT),
        HumanMessage(content=f"Technical Analysis: {ta_data}\n\nFundamental Analysis: {fa_data}\n\nRisk Assessment: {risk_data}")
    ]
    response = llm.invoke(messages)
    return {"final_recommendation": response.content}

# === 5. Graph Construction ===
def create_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("MarketData", market_data_node)
    workflow.add_node("TechnicalAnalyst", technical_analyst_node)
    workflow.add_node("FundamentalAnalyst", fundamental_analyst_node)
    workflow.add_node("RiskManager", risk_manager_node)
    workflow.add_node("StrategyGenerator", strategy_generator_node)

    # Add Edges
    workflow.set_entry_point("MarketData")

    # Parallel: Data -> TA & FA
    workflow.add_edge("MarketData", "TechnicalAnalyst")
    workflow.add_edge("MarketData", "FundamentalAnalyst")

    # Converge: TA & FA -> Risk
    workflow.add_edge("TechnicalAnalyst", "RiskManager")
    workflow.add_edge("FundamentalAnalyst", "RiskManager")

    # Sequence: Risk -> Strategy
    workflow.add_edge("RiskManager", "StrategyGenerator")

    # End
    workflow.add_edge("StrategyGenerator", END)

    return workflow.compile()

# Global app instance for import
app = create_graph()
