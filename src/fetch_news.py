import yfinance as yf
import pandas as pd
import os
from time import sleep

data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)

# Load cleaned CSV (only SYMBOL column)
tickers_csv = os.path.join(data_dir, "stockNames.csv")
tickers_df = pd.read_csv(tickers_csv)
tickers_df.columns = tickers_df.columns.str.strip()
stocks = tickers_df['SYMBOL'].tolist()

all_data = []

for stock in stocks:
    try:
        ticker_symbol = stock.strip() + ".NS"
        print(f"Fetching data for {ticker_symbol}...")

        df = yf.download(ticker_symbol, period="3y", interval="1d")
        if df.empty:
            print(f"No data for {stock}, skipping.")
            continue

        df.reset_index(inplace=True)

        # Keep only required columns
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[c for c in cols if c in df.columns]]

        # Add Ticker column
        df['Ticker'] = stock

        all_data.append(df)

        sleep(1)

    except Exception as e:
        print(f"Error fetching data for {stock}: {e}")

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(os.path.join(data_dir, "stocks_all.csv"), index=False)
    print(f"All stocks data saved! Rows: {len(combined_df)}")
else:
    print("No data fetched for any stock!")
