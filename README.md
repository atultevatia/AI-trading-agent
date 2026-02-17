# LangGraph AI Trading Agent: Setup Guide

This project implements an AI-powered trading agent for Indian Equities using **LangGraph** for orchestration and **Streamlit** for the UI.

## File Structure
- `app.py`: Main Streamlit application.
- `graph_code.py`: LangGraph logic, defining nodes (Market Data, TA, FA, Risk) and the workflow.
- `agent_prompts.md`: Detailed system prompts used by the agents.
- `requirements.txt`: Python dependencies.

## Prerequisites
- Python 3.9+
- OpenAI API Key (if you enable the real LLM calls in `graph_code.py`)

## Installation

1.  **Clone the repository** (or download files):
    Ensure `app.py`, `graph_code.py`, and `requirements.txt` are in the same folder.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup**:
    If using real OpenAI calls, create a `.env` file:
    ```bash
    OPENAI_API_KEY=sk-your-key-here
    ```

## Running the App

Run the Streamlit server:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## Customization
- **Enable Real LLMs**: Uncomment the `ChatOpenAI` code in `graph_code.py` to use real GPT-4 analysis instead of the mock responses.
- **Add Real Data**: Integrate `yfinance` in the `fetch_market_data` function in `graph_code.py`.
