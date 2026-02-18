import streamlit as st
import os
import pandas as pd
import json
import time
import yfinance as yf
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from sector_graph_code import (
    app, 
    ANALYST_AGENT_PROMPT, 
    RISK_MANAGER_PROMPT, 
    PORTFOLIO_MANAGER_PROMPT
)
import paper_trade_engine as engine

# --- Secrets Handling for Streamlit Cloud ---
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    pass # Running locally, will rely on .env via dotenv loaded in graph code

if "pending_ticker" not in st.session_state:
    st.session_state.pending_ticker = None

st.set_page_config(page_title="AI Sector Scanner", layout="wide")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "strategy_prompts" not in st.session_state:
    st.session_state.strategy_prompts = {
        "analyst": ANALYST_AGENT_PROMPT,
        "risk": RISK_MANAGER_PROMPT,
        "portfolio": PORTFOLIO_MANAGER_PROMPT
    }

st.title("🤖 Autonomous AI Trading Agent")
st.caption("Multi-Agent reasoning system: Analyst, Risk Manager, Portfolio Manager, Paper Trading & Custom Strategy")

# --- Tabs ---
tab_scanner, tab_portfolio, tab_strategy = st.tabs(["🔍 Sector Scanner", "📊 Paper Portfolio", "⚙️ Strategy Settings"])

# Sidebar
with st.sidebar:
    st.header("Agent Control")
    sector = st.selectbox("Market Sector", ["AI", "AUTO", "BANK", "PHARMA", "FMCG"])
    scan_btn = st.button("Activate Agent Loop")

with tab_scanner:
    if scan_btn:
        with st.spinner(f"Agent Loop active for {sector}... Analyzing data and news."):
            try:
                initial_state = {
                    "sector": sector, 
                    "tickers": [], 
                    "analyses": [], 
                    "portfolio": [], 
                    "remaining_cash": 10000,
                    "prompts": st.session_state.strategy_prompts
                }
                result = app.invoke(initial_state)
                st.session_state.last_result = result # Persist for interaction
                st.session_state.messages = [] # Reset chat on new scan
            except Exception as e:
                st.error(f"Agent Loop Failed: {e}")
                st.exception(e)

    # Always show the latest result if it exists in session state
    if st.session_state.last_result:
        res = st.session_state.last_result
        
        # --- Portfolio Allocation ---
        portfolio = res.get("portfolio", [])
        remaining = res.get("remaining_cash", 10000)
        
        st.header("🎯 Final Portfolio Allocation")
        if not portfolio:
            st.warning("The Portfolio Manager did not allocate capital to any trades in this scan cycle.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                df_portfolio = pd.DataFrame(portfolio)
                st.table(df_portfolio[['ticker', 'shares', 'weight_pct', 'reason']])
            with c2:
                st.metric("Remaining Cash", f"₹{remaining:,.2f}")
                st.metric("Total Stocks", len(portfolio))

        # --- Agent Reasoning Log ---
        st.markdown("---")
        st.header("🧠 Agent Reasoning Logs")
        
        analyses = res.get("analyses", [])
        if not analyses:
            st.info("No deep analysis logs found.")
        else:
            for pick in analyses:
                status_emoji = "✅" if pick.get('risk_status') == "APPROVED" else "❌"
                with st.expander(f"{status_emoji} REASONING: {pick['ticker']} (Conviction: {pick['conviction']}%)"):
                    col_a, col_r = st.columns(2)
                    
                    with col_a:
                        st.subheader("💡 Analyst Pitch")
                        st.write(pick['thesis'])
                        st.json({
                            "Entry": pick['entry'],
                            "Target": pick['target'],
                            "Stop": pick['stop_loss']
                        })
                        
                    with col_r:
                        st.subheader("🛡️ Risk Review")
                        if pick.get('risk_status') == "APPROVED":
                            st.info(pick['risk_criticism'])
                            st.success(f"Adjusted Stop: {pick['adjusted_stop']}")
                            st.write("**Risk Status:** APPROVED")
                            
                            # Book Trade Logic with Human-in-the-Loop Confirmation
                            allocation = next((a for a in portfolio if a['ticker'] == pick['ticker']), None)
                            qty = allocation['shares'] if allocation else 10
                            
                            if st.session_state.get("pending_ticker") == pick['ticker']:
                                st.warning(f"⚠️ **Review Trade: {qty} shares of {pick['ticker']} at ₹{pick['price']:.2f}**")
                                st.write(f"Stop Loss: ₹{pick['adjusted_stop']:.2f} | Target: ₹{pick['target']:.2f}")
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    if st.button("✅ Confirm Booking", key=f"confirm_{pick['ticker']}"):
                                        success, msg = engine.book_trade(
                                            pick['ticker'], pick['price'], qty, 
                                            pick['adjusted_stop'], pick['target'], pick['thesis']
                                        )
                                        if success: 
                                            st.success(msg)
                                            st.session_state.pending_ticker = None
                                            time.sleep(1)
                                            st.rerun()
                                        else: st.warning(msg)
                                with c2:
                                    if st.button("❌ Cancel", key=f"cancel_{pick['ticker']}"):
                                        st.session_state.pending_ticker = None
                                        st.rerun()
                            else:
                                if st.button(f"Book Trade: {pick['ticker']}", key=f"book_{pick['ticker']}"):
                                    st.session_state.pending_ticker = pick['ticker']
                                    st.rerun()
                        else:
                            st.warning(pick['risk_criticism'])
                            st.write("**Risk Status:** REJECTED")

        # --- Interactive Analysis Chatbot ---
        st.markdown("---")
        st.header("💬 Chat with Analysis Agent")
        
        for message in st.session_state.messages:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            with st.chat_message(role):
                content = message.content
                display_content = content
                booking_proposal = None
                
                if role == "assistant" and "PROPOSE_BOOK:" in content:
                    parts = content.split("PROPOSE_BOOK:")
                    display_content = parts[0].strip()
                    try:
                        booking_proposal = json.loads(parts[1].strip())
                    except: pass
                
                st.markdown(display_content)
                
                if booking_proposal:
                    if st.button(f"Book {booking_proposal['ticker']} (Chat Manual Override)", key=f"hist_chat_book_{booking_proposal['ticker']}_{hash(content)}"):
                        success, msg = engine.book_trade(
                            booking_proposal['ticker'], 
                            booking_proposal['price'], 
                            booking_proposal['shares'], 
                            booking_proposal['stop'], 
                            booking_proposal['target'], 
                            "Manual override via Chat Agent"
                        )
                        if success: st.success(msg)
                        else: st.warning(msg)

        if prompt := st.chat_input("Ask about the scan results..."):
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            context_data = {
                "sector": res.get("sector"),
                "analyses": res.get("analyses"),
                "portfolio": res.get("portfolio")
            }

            with st.chat_message("assistant"):
                llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
                # Updated System Prompt to allow overrides and systematic booking signals
                system_msg = SystemMessage(content=f"""You are a professional quant analyst. 
                Context of latest scan: {json.dumps(context_data)}
                
                USER GUIDELINES:
                1. Answer questions concisely based on the scan data.
                2. If the user wants to book a trade (even if REJECTED), you should facilitate it.
                3. To propose a booking, your response MUST end with the following exact format on a new line:
                   PROPOSE_BOOK: {{"ticker": "TICKER", "shares": QTY, "price": PRICE, "stop": STOP, "target": TARGET}}
                4. Use the recommended 'shares' from the portfolio allocation if available, otherwise default to a reasonable amount (e.g., 10).
                5. Use the 'price', 'adjusted_stop', and 'target' from the analysis logs.""")
                
                history = st.session_state.messages[-10:]
                response = llm.invoke([system_msg] + history)
                
                # Render response and check for booking signal
                content = response.content
                display_content = content
                booking_proposal = None
                
                if "PROPOSE_BOOK:" in content:
                    parts = content.split("PROPOSE_BOOK:")
                    display_content = parts[0].strip()
                    try:
                        booking_proposal = json.loads(parts[1].strip())
                    except: pass

                st.markdown(display_content)
                st.session_state.messages.append(AIMessage(content=content))

                if booking_proposal:
                    st.warning(f"🛡️ **Manual Override: Book {booking_proposal['shares']} shares of {booking_proposal['ticker']}?**")
                    if st.button(f"Confirm Chat Booking: {booking_proposal['ticker']}", key=f"chat_book_{time.time()}"):
                        success, msg = engine.book_trade(
                            booking_proposal['ticker'], 
                            booking_proposal['price'], 
                            booking_proposal['shares'], 
                            booking_proposal['stop'], 
                            booking_proposal['target'], 
                            "Manual override via Chat Agent"
                        )
                        if success: st.success(msg)
                        else: st.warning(msg)

# --- Paper Portfolio Tab ---
with tab_portfolio:
    st.header("📈 Active Paper Positions")
    trades = engine.load_trades()
    active_trades = trades.get('active', [])
    
    if not active_trades:
        st.info("No active trades. Scan a sector and 'Book' a recommendation to start!")
    else:
        active_tickers = [t['ticker'] for t in active_trades]
        try:
            # Simple price fetch
            prices_df = yf.download(active_tickers, period="1d", progress=False)['Close']
        except:
            prices_df = pd.DataFrame()

        portfolio_data = []
        total_pnl = 0
        
        for t in active_trades:
            try:
                # Robust extraction of latest price
                if isinstance(prices_df, pd.Series):
                    curr = prices_df.iloc[-1]
                elif t['ticker'] in prices_df.columns:
                    curr = prices_df[t['ticker']].iloc[-1]
                else:
                    curr = t['entry_price'] # fallback

                pnl = (curr - t['entry_price']) * t['quantity']
                pnl_pct = ((curr / t['entry_price']) - 1) * 100
                total_pnl += pnl
                
                portfolio_data.append({
                    "Ticker": t['ticker'],
                    "Entry": f"₹{t['entry_price']:.2f}",
                    "Current": f"₹{curr:.2f}",
                    "Qty": t['quantity'],
                    "P&L": f"₹{pnl:.2f}",
                    "P&L %": f"{pnl_pct:+.2f}%",
                    "Stop": t['stop_loss'],
                    "Target": t['target']
                })
            except: continue

        if portfolio_data:
            st.table(portfolio_data)
            st.metric("Total Unrealized P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
        
            with st.expander("🛠️ Manage Trades"):
                target_ticker = st.selectbox("Select Trade to Close", [t['ticker'] for t in active_trades])
                if st.button("Close Selected Trade"):
                    trade_to_close = next(t for t in active_trades if t['ticker'] == target_ticker)
                    exit_price = trade_to_close['entry_price'] # default
                    try:
                        exit_price = yf.Ticker(target_ticker).fast_info['last_price']
                    except: pass
                    
                    s, m = engine.close_trade(trade_to_close['id'], exit_price)
                    if s: 
                        st.success(m)
                        time.sleep(1)
                        st.rerun()

    st.markdown("---")
    st.header("📜 Trade History")
    closed = trades.get('closed', [])
    if closed:
        st.dataframe(pd.DataFrame(closed)[['ticker', 'entry_price', 'exit_price', 'quantity', 'pnl', 'pnl_pct', 'entry_time']])
    
    # --- Download Data Option ---
    st.markdown("---")
    if active_trades or closed:
        json_data = json.dumps(trades, indent=4)
        st.download_button(
            label="📥 Download All Trade Data (JSON)",
            data=json_data,
            file_name="paper_trades_backup.json",
            mime="application/json",
            help="Back up your trades! Streamlit Cloud storage is ephemeral and clears on restarts."
        )

# --- Strategy Settings Tab ---
with tab_strategy:
    st.header("⚙️ Agent Strategy Customization")
    st.write("Customize the 'brain' of each agent. Your changes take effect in the next scan cycle.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👨‍🔬 Analyst Persona")
        st.session_state.strategy_prompts["analyst"] = st.text_area(
            "Analyst Instructions", 
            value=st.session_state.strategy_prompts["analyst"], 
            height=300,
            help="Define how the Analyst should interpret technicals and news sentiment."
        )
        
        st.subheader("🛡️ Risk Manager Persona")
        st.session_state.strategy_prompts["risk"] = st.text_area(
            "Risk Manager Instructions", 
            value=st.session_state.strategy_prompts["risk"], 
            height=300,
            help="Define the criteria for trade approval and rejections."
        )

    with col2:
        st.subheader("💼 Portfolio Manager Persona")
        st.session_state.strategy_prompts["portfolio"] = st.text_area(
            "Portfolio Manager Instructions", 
            value=st.session_state.strategy_prompts["portfolio"], 
            height=300,
            help="Define capital allocation rules and diversification constraints."
        )
        
        st.markdown("---")
        if st.button("🔄 Reset to Default Strategy"):
            st.session_state.strategy_prompts = {
                "analyst": ANALYST_AGENT_PROMPT,
                "risk": RISK_MANAGER_PROMPT,
                "portfolio": PORTFOLIO_MANAGER_PROMPT
            }
            st.success("Prompts reset to system defaults!")
            st.rerun()

st.markdown("---")
st.markdown("*Disclaimer: Autonomous AI recommendations. Not financial advice.*")
