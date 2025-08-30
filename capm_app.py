#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import yfinance as yf
import streamlit as st

st.title("📈 Multi-Stock CAPM & Investment Calculator")

# ------------------ User Info ------------------
st.sidebar.header("User Info")
user_name = st.sidebar.text_input("Enter Your Name:")
if not user_name:
    st.warning("Please enter your name to proceed.")
    st.stop()

# ------------------ Data Source ------------------
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Data Source", ("Use Local CSV", "Upload CSV", "Yahoo Finance"))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (in %)", 0.0, 10.0, 2.0)

local_csv_path = "Final-50-stocks.csv"

# ------------------ Load Data ------------------
if data_source == "Use Local CSV":
    if os.path.exists(local_csv_path):
        data = pd.read_csv(local_csv_path)
        st.subheader("📄 Dataset Preview (Local CSV)")
        st.dataframe(data.head())
    else:
        st.error(f"Local CSV not found at {local_csv_path}")
        st.stop()

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Dataset Preview (Uploaded CSV)")
        st.dataframe(data.head())
    else:
        st.info("Please upload a CSV file to proceed.")
        st.stop()

elif data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", "AAPL")
    benchmark = st.sidebar.text_input("Enter Benchmark Ticker (e.g. ^GSPC)", "^GSPC")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
    
    st.subheader("Downloading Data...")
    data = yf.download([ticker, benchmark], start=start_date, end=end_date)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"].reset_index()
        data.rename(columns={ticker: "Stock", benchmark: "Benchmark"}, inplace=True)
    else:
        data = data.reset_index()
    st.subheader("📄 Dataset Preview (Yahoo Finance)")
    st.dataframe(data.head())

# ------------------ Auto-detect Columns ------------------
date_col = data.columns[0]
data[date_col] = pd.to_datetime(data[date_col])
data.set_index(date_col, inplace=True)

benchmark_col = "Benchmark" if "Benchmark" in data.columns else data.columns[0]
stock_cols = [col for col in data.columns if col != benchmark_col]

st.sidebar.header("Select Stocks for CAPM Analysis")
selected_stocks = st.sidebar.multiselect("Choose Stocks", stock_cols, default=stock_cols[:5])

if not selected_stocks:
    st.warning("Please select at least one stock to proceed.")
    st.stop()

close_prices = data[[benchmark_col] + selected_stocks]

# ------------------ Closing Prices Chart ------------------
st.write("### Closing Prices")
st.line_chart(close_prices)

# ------------------ Investment Inputs ------------------
st.sidebar.header("Investment Calculator")
st.sidebar.write("💰 Enter investment amount per stock:")

investment_amounts = {}
for stock in selected_stocks:
    investment_amounts[stock] = st.sidebar.number_input(f"Investment for {stock} (₹)", min_value=1000, value=100000, step=1000)

start_invest = st.sidebar.date_input(
    "Investment Start Date", 
    value=close_prices.index.min(), 
    min_value=close_prices.index.min(), 
    max_value=close_prices.index.max()
)
end_invest = st.sidebar.date_input(
    "Investment End Date", 
    value=close_prices.index.max(), 
    min_value=close_prices.index.min(), 
    max_value=close_prices.index.max()
)

# ------------------ CAPM & Investment Calculation ------------------
results = []

for stock in selected_stocks:
    returns = close_prices[[stock, benchmark_col]].pct_change().dropna()
    stock_returns = returns[stock]
    market_returns = returns[benchmark_col]
    
    X = sm.add_constant(market_returns)
    model = sm.OLS(stock_returns, X).fit()
    alpha, beta = model.params
    
    # Expected annual return
    market_return = market_returns.mean() * 252
    rf = risk_free_rate / 100
    expected_return = rf + beta * (market_return - rf)
    
    # Investment projection
    period_data = close_prices.loc[start_invest:end_invest, [stock]]
    num_days = len(period_data)
    period_return = (1 + expected_return) ** (num_days / 252) - 1
    future_value = investment_amounts[stock] * (1 + period_return)
    
    results.append({
        "Stock": stock,
        "Alpha": round(alpha, 4),
        "Beta": round(beta, 4),
        "Expected Annual Return": expected_return,
        "Investment Amount": investment_amounts[stock],
        "Period Return": period_return,
        "Projected Value": future_value
    })

results_df = pd.DataFrame(results)

# ------------------ Portfolio Summary ------------------
total_investment = results_df["Investment Amount"].sum()
total_projected = results_df["Projected Value"].sum()
total_gain = total_projected - total_investment
total_return_pct = total_gain / total_investment

portfolio_summary = pd.DataFrame({
    "Total Investment": [total_investment],
    "Total Projected Value": [total_projected],
    "Total Gain": [total_gain],
    "Overall Return (%)": [total_return_pct]
})

# ------------------ Search / Filter ------------------
st.sidebar.header("Search Stock Results")
search_stock = st.sidebar.selectbox(
    "Select Stock to View Results (optional)", 
    options=["All"] + selected_stocks, 
    index=0
)

results_df_formatted = results_df.copy()
results_df_formatted["Expected Annual Return"] = results_df_formatted["Expected Annual Return"].map("{:.2%}".format)
results_df_formatted["Period Return"] = results_df_formatted["Period Return"].map("{:.2%}".format)
results_df_formatted["Projected Value"] = results_df_formatted["Projected Value"].map("₹{:,.2f}".format)
results_df_formatted["Investment Amount"] = results_df_formatted["Investment Amount"].map("₹{:,.2f}".format)

if search_stock != "All":
    filtered_df = results_df_formatted[results_df_formatted["Stock"] == search_stock]
else:
    filtered_df = results_df_formatted

# ------------------ Display Results ------------------
st.subheader(f"📊 CAPM & Investment Results for {user_name}")
st.dataframe(filtered_df)

# Portfolio Summary
st.write("### 📌 Portfolio Summary")
portfolio_summary_formatted = portfolio_summary.copy()
portfolio_summary_formatted["Total Investment"] = portfolio_summary_formatted["Total Investment"].map("₹{:,.2f}".format)
portfolio_summary_formatted["Total Projected Value"] = portfolio_summary_formatted["Total Projected Value"].map("₹{:,.2f}".format)
portfolio_summary_formatted["Total Gain"] = portfolio_summary_formatted["Total Gain"].map("₹{:,.2f}".format)
portfolio_summary_formatted["Overall Return (%)"] = portfolio_summary_formatted["Overall Return (%)"].map("{:.2%}".format)

st.table(portfolio_summary_formatted)

# Bar charts
st.write("### Projected Investment Values (Per Stock)")
st.bar_chart(results_df.set_index("Stock")["Projected Value"])

st.write("### Portfolio Investment vs Projected Value")
st.bar_chart(pd.DataFrame({
    "Investment": [total_investment],
    "Projected Value": [total_projected]
}, index=["Portfolio"]))


# In[1]:





# In[5]:





# In[ ]:




