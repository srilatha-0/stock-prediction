import pandas as pd
import os

DATA_DIR = "../data"
INPUT_FILE = os.path.join(DATA_DIR, "stocks_all.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "stocks_all_preprocessed.csv")

df = pd.read_csv(INPUT_FILE)

# Ensure Date format
df['Date'] = pd.to_datetime(df['Date'])

# Sort properly
df = df.sort_values(['Ticker','Date'])

# --- Technical Indicators ---

# Moving Averages
df['MA10'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(10).mean())
df['MA20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())

# Exponential Moving Averages
df['EMA10'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
df['EMA20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())

# RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df['RSI'] = df.groupby('Ticker')['Close'].transform(compute_rsi)

# MACD
exp12 = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
exp26 = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df['MACD'] = exp12 - exp26

# Drop initial NaN rows from rolling calculations
df = df.dropna()

# Save
df.to_csv(OUTPUT_FILE, index=False)
print("Indicators computed → stocks_all_preprocessed.csv created")
