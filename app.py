import os
import time
import html
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from textblob import TextBlob
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -----------------------
# LOAD ENV VARIABLES
# -----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API = os.getenv("ALPHA_VANTAGE_KEY")

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="StockSense AI Pro",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# -----------------------
# THEME CSS
# -----------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #0d1117 !important;
    color: #e6edf3;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #238636, #2ea043);
    color: #fff !important;
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 600;
    margin-top: 6px;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
}
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #58a6ff;
    margin: 18px 0 10px 0;
    border-left: 4px solid #238636;
    padding-left: 10px;
}
.hero {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.hero h1 { margin: 0; color: #58a6ff; font-size: 32px; }
.hero p  { margin: 5px 0 0 0; color: #8b949e; }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}
.badge-positive { background: #1a4731; color: #3fb950; }
.badge-negative { background: #4d1a1a; color: #f85149; }
.badge-neutral  { background: #2d2a1a; color: #e3b341; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📈 StockSense AI Pro</h1>
    <p>Advanced stock analysis with technical indicators, news sentiment, and AI insight.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# SIDEBAR
# -----------------------
companies = {
    "Apple": "AAPL", "Tesla": "TSLA", "Microsoft": "MSFT", "Google": "GOOGL",
    "Amazon": "AMZN", "NVIDIA": "NVDA", "Meta": "META", "Netflix": "NFLX", "Bank of America": "BAC"
}

st.sidebar.markdown("## ⚙️ Settings")
mode = st.sidebar.radio("Choose mode:", ["Quick Select", "Manual Ticker"])

if mode == "Quick Select":
    company = st.sidebar.selectbox("🏢 Company", list(companies.keys()))
    symbol = companies[company]
else:
    symbol = st.sidebar.text_input("🔤 Ticker", value="AAPL").strip().upper()

period = st.sidebar.selectbox("📅 Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox("⏱️ Interval", ["1d", "1h", "30m"], index=0)
show_indicators = st.sidebar.checkbox("📐 Show technical indicators", value=True)
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh every 10s", value=False)

st.sidebar.markdown("---")
analyze = st.sidebar.button("🔍 Analyze Stock")
ai_analysis = st.sidebar.button("🤖 AI Investment Insight")

# -----------------------
# UTILITIES
# -----------------------
def humanize_number(n):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    n = float(n)
    abs_n = abs(n)
    if abs_n >= 1e12: return f"{n/1e12:.2f}T"
    if abs_n >= 1e9:  return f"{n/1e9:.2f}B"
    if abs_n >= 1e6:  return f"{n/1e6:.2f}M"
    if abs_n >= 1e3:  return f"{n/1e3:.2f}K"
    return f"{n:.0f}"

def classify_sentiment(text):
    pol = TextBlob(text).sentiment.polarity
    if pol > 0:
        return "Positive", "positive"
    if pol < 0:
        return "Negative", "negative"
    return "Neutral", "neutral"

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_MID"] = sma20
    df["BB_UPPER"] = sma20 + (2 * std20)
    df["BB_LOWER"] = sma20 - (2 * std20)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    df["RSI14"] = compute_rsi(df["Close"], 14)
    return df

@st.cache_data(ttl=60)
def fetch_stock(symbol, period, interval):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=False)
    info = ticker.info if ticker else {}
    return df, info

@st.cache_data(ttl=300)
def get_news(symbol, api_key, max_items=8):
    if not api_key:
        return []

    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
    )
    for _ in range(2):  # simple retry
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            data = {}
    else:
        return []

    articles = []
    for item in data.get("feed", [])[:max_items]:
        title = item.get("title", "No title")
        safe_title = html.escape(title)
        label, css_key = classify_sentiment(title)
        icon = "📈" if css_key == "positive" else ("📉" if css_key == "negative" else "⚖️")
        badge = f'<span class="badge badge-{css_key}">{icon} {label}</span>'
        source = html.escape(item.get("source", "Unknown"))
        url = item.get("url", "")
        safe_url = html.escape(url)
        articles.append({
            "Headline": safe_title,
            "Sentiment": badge,
            "Source": source,
            "URL": safe_url
        })
    return articles

# -----------------------
# MAIN ANALYSIS
# -----------------------
if analyze:
    with st.spinner("Fetching market data..."):
        df, info = fetch_stock(symbol, period, interval)

    if df is None or df.empty:
        st.error("❌ No data found. Try another ticker/period/interval.")
    else:
        df = add_indicators(df)

        # Metrics
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
        delta = last_close - prev_close
        delta_pct = (delta / prev_close * 100) if prev_close else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${last_close:.2f}", f"{delta:+.2f} ({delta_pct:+.2f}%)")
        c2.metric("Period High", f"${df['High'].max():.2f}")
        c3.metric("Period Low", f"${df['Low'].min():.2f}")
        c4.metric("Last Volume", humanize_number(df["Volume"].iloc[-1]))

        # Overview metrics
        st.markdown('<div class="section-title">🏢 Company Overview</div>', unsafe_allow_html=True)
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Market Cap", humanize_number(info.get("marketCap")))
        o2.metric("P/E Ratio", f"{info.get('trailingPE'):.2f}" if isinstance(info.get("trailingPE"), (float, int)) else "N/A")
        o3.metric("52W High", f"${info.get('fiftyTwoWeekHigh'):.2f}" if isinstance(info.get("fiftyTwoWeekHigh"), (float, int)) else "N/A")
        o4.metric("52W Low", f"${info.get('fiftyTwoWeekLow'):.2f}" if isinstance(info.get("fiftyTwoWeekLow"), (float, int)) else "N/A")

        tabs = st.tabs(["📊 Overview", "📐 Technicals", "📰 News", "📋 Data"])

        # Overview tab
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="Candles"
            ))
            if show_indicators:
                fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color="#f59e0b", width=1.5)))
                fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color="#22c55e", width=1.5)))
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], name="BB Upper", line=dict(color="#64748b", width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], name="BB Lower", line=dict(color="#64748b", width=1, dash="dot")))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h"),
                height=520
            )
            st.plotly_chart(fig, use_container_width=True)

            vol_fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], marker_color="rgba(46,160,67,0.6)", name="Volume"))
            vol_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", height=220, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(vol_fig, use_container_width=True)

        # Technicals tab
        with tabs[1]:
            r1, r2 = st.columns(2)
            with r1:
                rsi_fig = go.Figure(go.Scatter(x=df.index, y=df["RSI14"], line=dict(color="#a78bfa", width=2), name="RSI14"))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                rsi_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", height=280, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(rsi_fig, use_container_width=True)

            with r2:
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#60a5fa", width=2)))
                macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#f59e0b", width=2)))
                macd_fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Histogram", marker_color="rgba(16,185,129,0.5)"))
                macd_fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", height=280, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(macd_fig, use_container_width=True)

            latest_rsi = df["RSI14"].iloc[-1]
            if pd.notna(latest_rsi):
                if latest_rsi > 70:
                    st.warning(f"RSI14 = {latest_rsi:.1f} → potentially overbought.")
                elif latest_rsi < 30:
                    st.info(f"RSI14 = {latest_rsi:.1f} → potentially oversold.")
                else:
                    st.success(f"RSI14 = {latest_rsi:.1f} → neutral zone.")

        # News tab
        with tabs[2]:
            with st.spinner("Fetching latest news..."):
                articles = get_news(symbol, NEWS_API)

            if not NEWS_API:
                st.warning("Set ALPHA_VANTAGE_KEY in .env to enable news sentiment.")
            elif not articles:
                st.warning("No recent news or API limit reached.")
            else:
                rows = []
                for a in articles:
                    link = f"<a href='{a['URL']}' target='_blank' style='color:#58a6ff;'>Open</a>" if a["URL"] else "-"
                    rows.append(
                        f"<tr>"
                        f"<td>{a['Headline']}</td>"
                        f"<td style='text-align:center'>{a['Sentiment']}</td>"
                        f"<td>{a['Source']}</td>"
                        f"<td style='text-align:center'>{link}</td>"
                        f"</tr>"
                    )
                table_html = (
                    "<table><thead><tr>"
                    "<th>Headline</th><th>Sentiment</th><th>Source</th><th>Link</th>"
                    "</tr></thead><tbody>"
                    + "".join(rows) +
                    "</tbody></table>"
                )
                st.markdown(table_html, unsafe_allow_html=True)

        # Data tab
        with tabs[3]:
            view_df = df.tail(100).copy().reset_index()
            st.dataframe(
                view_df.style.format({
                    "Open": "${:.2f}", "High": "${:.2f}", "Low": "${:.2f}", "Close": "${:.2f}",
                    "EMA20": "${:.2f}", "EMA50": "${:.2f}",
                    "BB_UPPER": "${:.2f}", "BB_LOWER": "${:.2f}",
                    "Volume": "{:,.0f}", "RSI14": "{:.2f}", "MACD": "{:.3f}", "MACD_SIGNAL": "{:.3f}", "MACD_HIST": "{:.3f}",
                }),
                use_container_width=True
            )
            csv_data = view_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv_data, file_name=f"{symbol}_{period}_{interval}.csv", mime="text/csv")

# -----------------------
# AI ANALYSIS
# -----------------------
if ai_analysis:
    st.markdown('<div class="section-title">🤖 AI Investment Insight</div>', unsafe_allow_html=True)

    try:
        df_ai, _ = fetch_stock(symbol, period, interval)
        if df_ai.empty:
            st.error("No stock data available for AI context.")
        else:
            df_ai = add_indicators(df_ai)
            latest = df_ai.iloc[-1]

            context = {
                "close": round(float(latest["Close"]), 2),
                "ema20": round(float(latest["EMA20"]), 2) if pd.notna(latest["EMA20"]) else None,
                "ema50": round(float(latest["EMA50"]), 2) if pd.notna(latest["EMA50"]) else None,
                "rsi14": round(float(latest["RSI14"]), 2) if pd.notna(latest["RSI14"]) else None,
                "macd": round(float(latest["MACD"]), 4) if pd.notna(latest["MACD"]) else None,
                "signal": round(float(latest["MACD_SIGNAL"]), 4) if pd.notna(latest["MACD_SIGNAL"]) else None,
                "high_period": round(float(df_ai["High"].max()), 2),
                "low_period": round(float(df_ai["Low"].min()), 2),
            }

            with st.spinner("Generating AI analysis..."):
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama-3.3-70b-versatile",
                )
                prompt = f"""
You are a professional financial analyst.

Analyze ticker: {symbol}
Time period: {period}
Interval: {interval}
Latest stats: {context}

Deliver:
1) Trend Summary
2) Bull vs Bear case
3) Risk factors
4) Key levels (support/resistance)
5) Short-term vs long-term outlook
6) Action checklist for retail investor

Keep it concise, practical, and balanced.
Do not provide guaranteed returns.
"""
                response = llm.invoke(prompt)
                st.markdown(
                    f"""
                    <div style="background:#161b22;border:1px solid #238636;border-radius:12px;padding:18px;line-height:1.7;white-space:pre-wrap;">
                    {html.escape(response.content)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.error(f"AI analysis failed: {e}")

# -----------------------
# AUTO REFRESH
# -----------------------
if auto_refresh:
    st.caption("Auto-refresh is ON (10s)")
    time.sleep(10)
    st.rerun()