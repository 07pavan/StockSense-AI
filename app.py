import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import requests
from langchain_groq import ChatGroq

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="StockSense AI",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS (Professional UI)
# ----------------------------

st.markdown("""
<style>

.stApp {
background-image: url("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3");
background-size: cover;
background-position: center;
background-attachment: fixed;
}

/* Dark overlay */
.main {
background-color: rgba(0,0,0,0.65);
padding: 20px;
border-radius: 10px;
}

/* Title */
h1 {
color: white;
}

/* Sub headers */
h2, h3 {
color: #f5f5f5;
}

/* Paragraph text */
p {
color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
background-color: #0e1117;
}

/* Metrics */
[data-testid="metric-container"] {
background-color: rgba(255,255,255,0.05);
padding: 15px;
border-radius: 10px;
border: 1px solid rgba(255,255,255,0.1);
}

/* AI output box */
.ai-box {
background-color: rgba(0,0,0,0.75);
padding: 20px;
border-radius: 12px;
border: 1px solid #00c3ff;
color: white;
font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------
# TITLE
# ----------------------------

st.title("📈 StockSense AI")
st.write("AI-powered stock analysis with market data and intelligent insights")

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("⚙ Settings")

groq_key = st.sidebar.text_input("Groq API Key", type="password")

symbol = st.sidebar.text_input("Stock Symbol", "AAPL")

period = st.sidebar.selectbox(
"Select Time Period",
["1mo","3mo","6mo","1y","5y"]
)

analyze_btn = st.sidebar.button("🔍 Analyze Stock")
ai_btn = st.sidebar.button("🤖 AI Investment Insight")

# ----------------------------
# FETCH STOCK DATA
# ----------------------------

def fetch_stock(symbol, period):

    stock = yf.Ticker(symbol)
    df = stock.history(period=period)

    return df


# ----------------------------
# SENTIMENT
# ----------------------------

def sentiment(text):

    analysis = TextBlob(text)

    if analysis.sentiment.polarity > 0:
        return "Positive 📈"

    elif analysis.sentiment.polarity < 0:
        return "Negative 📉"

    else:
        return "Neutral"


# ----------------------------
# NEWS API
# ----------------------------

def get_news(symbol):

    API_KEY = ""

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


# ----------------------------
# ANALYZE STOCK
# ----------------------------

if analyze_btn:

    df = fetch_stock(symbol, period)

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"${round(df['Close'].iloc[-1],2)}")

    col2.metric("Highest Price", f"${round(df['High'].max(),2)}")

    col3.metric("Lowest Price", f"${round(df['Low'].min(),2)}")

    st.divider()

    # -----------------------
    # STOCK CHART
    # -----------------------

    st.subheader("📊 Stock Price Chart")

    fig = px.line(
        df,
        x=df.index,
        y="Close",
        title=f"{symbol} Price Trend"
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # DATA TABLE
    # -----------------------

    st.subheader("📋 Stock Data")

    st.dataframe(df.tail())

    # -----------------------
    # NEWS
    # -----------------------

    st.subheader("📰 News Sentiment")

    news = get_news(symbol)

    if not news.empty:
        st.table(news)
    else:
        st.warning("No news available")


# ----------------------------
# AI ANALYSIS
# ----------------------------

if ai_btn and groq_key:

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

    st.subheader("🤖 AI Investment Insight")

    st.markdown(
        f'<div class="ai-box">{response.content}</div>',
        unsafe_allow_html=True
    )