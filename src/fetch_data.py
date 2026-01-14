import yfinance as yf
import pandas as pd
import os

# list of stock tickers (example)
stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# folder to save data
data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)

# fetch historical data
for ticker in stocks:
    df = yf.download(ticker, period="3y", interval="1d")  # last 3 years daily data
    df.to_csv(f"{data_dir}/{ticker}_historical.csv")
    print(f"{ticker} data saved!")
