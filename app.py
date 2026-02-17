import streamlit as st
from graph_code import app, AgentState

st.set_page_config(page_title="Indian Equities AI Trader", layout="wide")

st.title("🇮🇳 AI Trading Agent (LangGraph)")
st.caption("Focus: AI & Auto Sectors | NSE/BSE")

# Sidebar for Inputs
with st.sidebar:
    st.header("Trade Settings")
    ticker = st.text_input("Enter Stock Ticker (e.g., TATAMOTORS.NS)", value="TATAMOTORS.NS")
    sector = st.selectbox("Sector", ["Auto", "AI/Tech", "Other"])
    analyze_btn = st.button("Analyze Stock")

# Main Analysis Area
if analyze_btn:
    with st.spinner(f"Running AI Agents on {ticker}..."):
        try:
            # Initialize State
            initial_state = {
                "ticker": ticker,
                "sector": sector,
                "messages": []
            }
            
            # Run the Graph
            result = app.invoke(initial_state)
            
            # Display Results using Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Strategy", "Technical", "Fundamental", "Risk"])
            
            with tab1:
                st.subheader("🚀 Final Trading Strategy")
                st.markdown(result.get("final_recommendation", "No recommendation generated."))
                
            with tab2:
                st.subheader("📈 Technical Analysis")
                st.markdown(result.get("technical_analysis", "No analysis."))
                
            with tab3:
                st.subheader("🏢 Fundamental Analysis")
                st.markdown(result.get("fundamental_analysis", "No analysis."))
                
            with tab4:
                st.subheader("🛡️ Risk Assessment")
                st.markdown(result.get("risk_assessment", "No assessment."))
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("*Disclaimer: This is an AI-generated analysis not financial advice.*")
