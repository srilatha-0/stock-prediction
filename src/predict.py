import pandas as pd
import joblib
import os

DATA_DIR = "../data"
MODEL_DIR = "../models"
OUTPUT_DIR = "../data"

MODEL_PATH = os.path.join(MODEL_DIR, "intraday_xgboost.pkl")
DATA_PATH = os.path.join(DATA_DIR, "stocks_all_preprocessed.csv")
SENTIMENT_PATH = os.path.join(DATA_DIR, "news_sentiment.csv")

# Load model
model = joblib.load(MODEL_PATH)

# Load data
df_stock = pd.read_csv(DATA_PATH)
df_sent = pd.read_csv(SENTIMENT_PATH)

df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_sent['Date'] = pd.to_datetime(df_sent['Date'])

# Merge sentiment
df = pd.merge(df_stock, df_sent[['Date','Stock','Sentiment']],
              left_on=['Date','Ticker'],
              right_on=['Date','Stock'],
              how='left')

df['Sentiment'] = df['Sentiment'].fillna(0).infer_objects(copy=False)

# Take latest record of each stock
latest_df = df.sort_values(['Ticker','Date']).groupby('Ticker').tail(1)

# Features
feature_cols = [
    'Open','High','Low','Close','Volume',
    'MA10','MA20','EMA10','EMA20','RSI','MACD','Sentiment'
]

X_latest = latest_df[feature_cols]

# Predict probability of Up
proba = model.predict_proba(X_latest)[:,1]

latest_df['Up_Probability'] = proba

# Rank Top 15
top15 = latest_df[['Ticker','Up_Probability']].sort_values(
    by='Up_Probability', ascending=False).head(15)

# Save predictions log
output_file = os.path.join(OUTPUT_DIR, "daily_predictions.csv")
top15.to_csv(output_file, index=False)

print("\nTop 15 Predicted Intraday Gainers:\n")
print(top15)

print("\nPredictions saved to data/daily_predictions.csv")
