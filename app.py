import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
from dotenv import load_dotenv
import openai

# ====== 載入 OpenAI API Key（之後可補）======
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    openai.api_key = api_key

# ====== 頁面設定 ======
st.set_page_config(page_title="ETF風險視覺化平台", layout="wide")

st.title("📊 ETF風險視覺化平台 Prototype")
st.markdown("選擇ETF以查看歷史走勢與AI生成摘要。")

# ====== 選單區 ======
etf_list = ["SPY", "QQQ", "VTI", "SSO", "DIA", "ARKK"]
selected_etf = st.selectbox("請選擇ETF：", etf_list)

# ====== 抓資料 ======
@st.cache_data(ttl=3600)
def load_data(ticker):
    data = yf.download(ticker, start="2020-01-01")
    data["Return"] = data["Close"].pct_change()
    return data

data = load_data(selected_etf)

# ====== 視覺化 ======
st.subheader(f"{selected_etf} 近年收盤價走勢")
# ====== 修正版：處理多層欄位情況 ======
if isinstance(data.columns, pd.MultiIndex):
    close_col = data["Close"][selected_etf]
else:
    close_col = data["Close"]

fig = px.line(x=data.index, y=close_col, title=f"{selected_etf} 收盤價走勢")
fig.update_layout(xaxis_title="日期", yaxis_title="收盤價（美元）")
st.plotly_chart(fig, use_container_width=True)

# ====== 簡易風險指標 ======
if len(data) > 0:
    volatility = float(data["Return"].std() * (252 ** 0.5) * 100)  # 年化波動率
    max_drawdown = float(((data["Close"] / data["Close"].cummax()) - 1).min() * 100)
    st.metric(label="📉 年化波動率 (%)", value=f"{volatility:.2f}")
    st.metric(label="⚠️ 最大回撤 (%)", value=f"{max_drawdown:.2f}")

# ====== AI 生成摘要（選擇性功能）======
if api_key:
    st.subheader("🤖 AI摘要解釋")
    user_prompt = f"請用一般人能懂的方式說明這檔ETF（{selected_etf}）的特性、風險與投資重點，不要給投資建議。"
    if st.button("產生AI摘要"):
        with st.spinner("AI分析中..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.7
            )
            st.write(response["choices"][0]["message"]["content"])
else:
    st.info("若要使用AI摘要功能，請在專案資料夾建立 `.env` 檔，並加入你的 OPENAI_API_KEY。")

# ====== 結尾 ======
st.markdown("---")
st.caption("ETF資料來源：Yahoo Finance | 本頁僅供教育用途，非投資建議。")
