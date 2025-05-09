# 📊 Finance AI Dashboard

An intelligent financial analysis dashboard that uses machine learning and NLP to forecast market prices and analyze financial news sentiment.

## 🚀 Features

- Time-series price forecasting for stocks or cryptocurrencies (LSTM / regression).
- Sentiment analysis of financial news (from RSS or API).
- Interactive dashboard (Dash + Plotly) to visualize prices, sentiment, and predictions.
- Article summarization using LangChain and OpenAI.
- REST API (FastAPI) for exposing prediction and sentiment results.

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/finance-ai-dashboard.git
   cd finance-ai-dashboard
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data models
   ```

5. Add your AAPL_prices.csv to the data directory

---

## 📁 Project Structure

```
finance-ai-dashboard/
├── data/               # Market data and news (CSV/API sources)
├── notebooks/          # Exploratory ML/NLP work in Jupyter
├── models/             # Trained ML and NLP models
├── app/                # Dash dashboard and FastAPI backend
│   ├── dashboard.py
│   └── api.py
├── nlp/                # Sentiment analysis and summarization logic
│   ├── sentiment.py
│   └── summarizer.py
├── timeseries/         # Forecasting models for time series analysis
│   ├── predictor.py    # time series predictions
│   └── data_loader.py  # load data using Yahoo Finance API
├── requirements.txt
└── README.md
```

---

## 🔧 Technologies Used

- **Python 3.10+**
- **TensorFlow / PyTorch** – for price prediction (e.g., LSTM)
- **Scikit-learn** – for sentiment classification
- **LangChain + OpenAI API** – for summarizing financial articles
- **Plotly / Dash** – for interactive data visualization
- **FastAPI** – REST API backend
- **SQLite / PostgreSQL** – for data storage

---

## ✅ Tasks To Complete

1. [x] Fetch historical price data from Yahoo Finance or Binance API ✅ `timeseries/data_loader.py`
2. [x] Build a forecasting model (LSTM or regression)
3. [ ] Implement sentiment analysis from financial news
4. [ ] Train sentiment classifier (e.g., TF-IDF + LogisticRegression)
5. [ ] Build dashboard using Dash and Plotly
6. [ ] Add REST API using FastAPI
7. [ ] Use LangChain to generate article summaries

---

## 📌 Sample Input Data

- Time-series data: CSV with columns `date`, `close_price`, `volume`
- News: list of headlines and article texts with timestamps

---

## 📈 Final Output

- Web dashboard showing:
  - historical and predicted prices
  - news sentiment trends
  - key topics and summaries

---

## ✨ References & Tools

- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [NewsAPI](https://newsapi.org/)
- [LangChain Docs](https://docs.langchain.com/)
- [Dash by Plotly](https://dash.plotly.com/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 🧠 Author: Mateusz Leszko
Created as part of a career shift into Python AI Development. Contributions and feedback are welcome via issues or pull requests!