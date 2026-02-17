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

st.title("🤖 AI Sector Scanner (LangGraph)")
st.caption("Scan entire sectors for high-probability setups | NSE/BSE")

# Sidebar
with st.sidebar:
    st.header("Scanner Settings")
    sector = st.selectbox("Select Sector", ["AI", "AUTO", "BANK", "PHARMA", "FMCG"])
    scan_btn = st.button("Start Scan")


# Main Area
if scan_btn:
    with st.spinner(f"Scanning {sector} Sector... This may take a minute."):
        try:
            initial_state = {"sector": sector, "tickers": [], "results": [], "final_ranking": []}
            result = app.invoke(initial_state)
            
            # --- Results Display ---
            final_picks = result.get("final_ranking", [])
            
            if not final_picks:
                st.warning("No stocks met the filter criteria.")
            else:
                st.success(f"Analyzed {len(final_picks)} opportunities!")
                
                # Separate by Tier
                strong_buys = [p for p in final_picks if p['tier'] == 'STRONG_BUY']
                watchlist = [p for p in final_picks if p['tier'] == 'WATCHLIST']
                monitor = [p for p in final_picks if p['tier'] == 'MONITOR']
                
                # --- Tabbed View ---
                t1, t2, t3 = st.tabs([
                    f"🚀 Strong Buy ({len(strong_buys)})", 
                    f"👀 Watchlist ({len(watchlist)})", 
                    f"📡 Monitor ({len(monitor)})"
                ])
                
                def display_tier(picks, color_help):
                    if not picks:
                        st.info("No stocks in this tier.")
                        return
                        
                    for pick in picks:
                        with st.expander(f"{pick['ticker']} | Conf: {pick['confidence']:.2f}"):
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Price", pick['price'])
                            c2.metric("Target", pick['target'])
                            c3.metric("Stop Loss", pick['stop'])
                            c4.metric("Tech Score", round(pick.get('tech_score', 0), 2))
                            
                            st.markdown(f"**Reasoning:** {pick['reasoning']}")
                            if color_help == "green":
                                st.success(f"Rec. Size: {pick['position_size']} (Risk: 1%)")
                            elif color_help == "yellow":
                                st.warning("Watch for entry trigger.")
                            else:
                                st.info("Keep on radar.")

                with t1:
                    display_tier(strong_buys, "green")
                with t2:
                    display_tier(watchlist, "yellow")
                with t3:
                    display_tier(monitor, "blue")
                        
        except Exception as e:
            st.error(f"Scan failed: {e}")

st.markdown("---")
st.markdown("*Disclaimer: AI-generated analysis based on simulated mock data/news. Not financial advice.*")
