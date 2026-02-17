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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="AI Sector Scanner", layout="wide")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

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
            st.session_state.last_result = result # Persist for chatbot
            st.session_state.messages = [] # Reset chat on new scan
            
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
                            else:
                                st.warning(pick['risk_criticism'])
                                st.write("**Risk Status:** REJECTED")


        except Exception as e:
            st.error(f"Agent Loop Failed: {e}")
            st.exception(e)

import json

st.markdown("---")

# --- Interactive Analysis Chatbot ---
if st.session_state.last_result:
    st.header("💬 Chat with Analysis Agent")
    st.caption("Ask questions about rejections, convictions, or specific stock setups from the scan above.")

    # Display chat messages from history
    for message in st.session_state.messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

    if prompt := st.chat_input("Ask about SBI or TATA MOTORS..."):
        # Add user message to state
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare context from last result
        context_data = {
            "sector": st.session_state.last_result.get("sector"),
            "analyses": st.session_state.last_result.get("analyses"),
            "portfolio": st.session_state.last_result.get("portfolio")
        }

        # Generate Agent Response
        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
            system_msg = SystemMessage(content=f"""You are the Analysis AI Agent. 
            You have just completed a sector scan for {context_data['sector']}.
            
            Context of latest scan:
            {json.dumps(context_data, indent=2)}
            
            Your task is to answer user queries about the stocks, rejections, and portfolio choices based ONLY on the data above. 
            If asked about a stock not in the list, explain that it wasn't part of this scan cycle.
            Be concise and act like a professional quantitative analyst.""")
            
            # Context window management (last 10 messages)
            history = st.session_state.messages[-10:]
            response = llm.invoke([system_msg] + history)
            
            st.markdown(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))

st.markdown("---")
st.markdown("*Disclaimer: Fully autonomous AI trading recommendations. Not financial advice. Past performance is no guarantee of future results.*")

