# Implement LangGraph Sector Scanning Agent

- [x] Design `sector_graph_code.py` with Map-Reduce Architecture <!-- id: 0 -->
    - [x] Define `OverallState` and `StockState` schemas
    - [x] Create `SectorLoader` node (Ticker Mapping)
    - [x] Create `StockAnalyst` node (TA + Sentiment + Risk)
    - [x] Create `Ranker` node (Filter + sort top 5)
- [x] Implement robust `yfinance` data fetching with error handling <!-- id: 1 -->
- [x] Update `app.py` to support Sector Input and Table Output <!-- id: 2 -->
- [x] Verify functionality with "AI" and "AUTO" sectors <!-- id: 3 -->
- [x] Implement Tiered Decision Logic (Strong/Watch/Monitor) in `sector_graph_code.py` <!-- id: 4 -->
- [x] Update `app.py` to display Tiers and Confidence <!-- id: 5 -->
- [x] Integrate Live NSE Sector Data (Replace hardcoded map) <!-- id: 6 -->
    - [x] Research python libraries for NSE indices (`nselib` or `yfinance` ^CNXAUTO etc)
    - [x] Update `SectorLoader` node to fetch dynamic constituents
    - [x] Update `app.py` sector selection dropdown

