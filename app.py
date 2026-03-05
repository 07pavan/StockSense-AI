import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import requests
from langchain_groq import ChatGroq

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Stock Analyst", layout="wide")

st.title("📈 AI Stock Analyst")
st.write("Analyze stock prices, news sentiment, and AI insights")

# -----------------------------
# GROQ API KEY
# -----------------------------
groq_key = st.text_input("Enter Groq API Key", type="password")

# -----------------------------
# STOCK INPUT
# -----------------------------
symbol = st.text_input("Enter Stock Symbol", "AAPL")

# -----------------------------
# FETCH STOCK DATA
# -----------------------------
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="6mo")
    return df

# -----------------------------
# PLOT STOCK DATA
# -----------------------------
def plot_stock_data(df):

    fig = px.line(
        df,
        x=df.index,
        y="Close",
        title="Stock Price (Last 6 Months)"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SENTIMENT ANALYSIS
# -----------------------------
def sentiment(text):

    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0:
        return "Positive 😊"
    elif analysis.sentiment.polarity < 0:
        return "Negative 😟"
    else:
        return "Neutral 😐"

# -----------------------------
# GET NEWS
# -----------------------------
def get_news(symbol):

    API_KEY = "DOKL1FMFTXUVXRA1"

    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={API_KEY}"

    r = requests.get(url)
    data = r.json()

    articles = []

    if "feed" in data:
        for item in data["feed"][:5]:

            title = item["title"]
            senti = sentiment(title)

            articles.append({
                "Headline": title,
                "Sentiment": senti
            })

    return pd.DataFrame(articles)

# -----------------------------
# ANALYZE STOCK BUTTON
# -----------------------------
if st.button("Analyze Stock"):

    df = fetch_stock_data(symbol)

    st.subheader("📊 Stock Data")
    st.dataframe(df.tail())

    st.subheader("📈 Stock Chart")
    plot_stock_data(df)

    st.subheader("📰 News Sentiment")

    news = get_news(symbol)

    if not news.empty:
        st.table(news)
    else:
        st.warning("No news found")

# -----------------------------
# AI ANALYSIS
# -----------------------------
if groq_key and st.button("AI Analysis"):

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.3-70b-versatile"
    )

    prompt = f"""
    Analyze the stock {symbol}.

    Provide:
    1. Trend summary
    2. Investment insight
    3. Risk analysis
    """

    response = llm.invoke(prompt)

    st.subheader("🤖 AI Stock Analysis")
    st.write(response.content)