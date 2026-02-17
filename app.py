import streamlit as st
import os
import pandas as pd

# --- Secrets Handling for Streamlit Cloud ---
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    pass # Running locally, will rely on .env via dotenv loaded in graph code

from sector_graph_code import app

st.set_page_config(page_title="AI Sector Scanner", layout="wide")

st.title("🤖 Autonomous AI Trading Agent")
st.caption("Multi-Agent reasoning system: Analyst, Risk Manager, and Portfolio Manager")

# Sidebar
with st.sidebar:
    st.header("Agent Control")
    sector = st.selectbox("Market Sector", ["AI", "AUTO", "BANK", "PHARMA", "FMCG"])
    scan_btn = st.button("Activate Agent Loop")

# Main Area
if scan_btn:
    with st.spinner(f"Agent Loop active for {sector}... Analyzing data and news."):
        try:
            initial_state = {
                "sector": sector, 
                "tickers": [], 
                "analyses": [], 
                "portfolio": [], 
                "remaining_cash": 10000
            }
            result = app.invoke(initial_state)
            
            # --- Portfolio Allocation ---
            portfolio = result.get("portfolio", [])
            remaining = result.get("remaining_cash", 10000)
            
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
            
            analyses = result.get("analyses", [])
            if not analyses:
                st.info("No deep analysis logs found.")
            else:
                for pick in analyses:
                    with st.expander(f"REASONING: {pick['ticker']} (Conviction: {pick['conviction']}%)"):
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
                            st.info(pick['risk_criticism'])
                            st.success(f"Adjusted Stop: {pick['adjusted_stop']}")
                            st.write("**Risk Status:** APPROVED")

        except Exception as e:
            st.error(f"Agent Loop Failed: {e}")
            st.exception(e)

st.markdown("---")
st.markdown("*Disclaimer: Fully autonomous AI trading recommendations. Not financial advice. Past performance is no guarantee of future results.*")

