import pandas as pd
import os

# -----------------------
# Paths
# -----------------------
DATA_DIR = "../data"
INPUT_CSV = os.path.join(DATA_DIR, "stocks_all.csv")  # <-- fixed file name
OUTPUT_CSV = os.path.join(DATA_DIR, "stocks_all_preprocessed.csv")

# -----------------------
# Load cleaned stock data
# -----------------------
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"CSV file not found: {INPUT_CSV}. Make sure it exists in data/ folder.")

df = pd.read_csv(INPUT_CSV)

# Convert Date to datetime and sort
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

# -----------------------
# Compute technical indicators
# -----------------------
all_tickers = df['Ticker'].unique()
processed_data = []

for stock in all_tickers:
    df_stock = df[df['Ticker'] == stock].copy()
    df_stock = df_stock.sort_values(by='Date').reset_index(drop=True)

    # 1️⃣ Moving Averages
    df_stock['MA10'] = df_stock['Close'].rolling(window=10).mean()
    df_stock['MA50'] = df_stock['Close'].rolling(window=50).mean()

    # 2️⃣ Exponential Moving Average
    df_stock['EMA10'] = df_stock['Close'].ewm(span=10, adjust=False).mean()

    # 3️⃣ Daily Return
    df_stock['Return'] = df_stock['Close'].pct_change()

    # 4️⃣ Relative Strength Index (RSI)
    delta = df_stock['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_stock['RSI'] = 100 - (100 / (1 + rs))

    # 5️⃣ Volume Change
    df_stock['Volume_Change'] = df_stock['Volume'].pct_change()

    # Drop initial rows with NaN due to indicators
    df_stock = df_stock.dropna().reset_index(drop=True)

    processed_data.append(df_stock)

# -----------------------
# Combine all tickers
# -----------------------
final_df = pd.concat(processed_data, ignore_index=True)

# Save preprocessed CSV
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Preprocessed stock data saved: {OUTPUT_CSV}")
print(f"Columns: {final_df.columns.tolist()}")
print(f"Rows: {len(final_df)}")
