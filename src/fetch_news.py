import finnhub
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------
# Load API key from .env
# -----------------------------
load_dotenv()
api_key = os.getenv("FINNHUB_API_KEY")
if not api_key:
    raise ValueError("API key not found. Add FINNHUB_API_KEY to your .env file.")

# Initialize Finnhub client
client = finnhub.Client(api_key=api_key)

# -----------------------------
# Configuration
# -----------------------------
stocks = ["RELIANCE", "TCS", "INFY"]  # list of stock tickers/companies
data_dir = "../data"                   # folder to save CSV
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "news_headlines.csv")

# Load existing CSV if exists
if os.path.exists(file_path):
    df_existing = pd.read_csv(file_path)
else:
    df_existing = pd.DataFrame(columns=["Date", "Stock", "Headline", "Timestamp"])

# -----------------------------
# Fetch news per stock
# -----------------------------
news_data = []
today = datetime.today()
from_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")  # last 2 days
to_date = today.strftime("%Y-%m-%d")

for stock in stocks:
    try:
        # Fetch latest company news
        articles = client.company_news(stock, _from=from_date, to=to_date)

        # Fallback to general news if no company news
        if not articles:
            print(f"No company news for {stock}, falling back to general news.")
            articles = client.general_news('general', min_id=0)

        for article in articles[:5]:  # first 5 headlines
            headline = article.get("headline", "").strip()
            if headline:
                news_data.append({
                    "Date": today.strftime("%Y-%m-%d"),
                    "Stock": stock,
                    "Headline": headline,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        print(f"Fetched {len(articles[:5])} headlines for {stock}")

    except Exception as e:
        print(f"Error fetching news for {stock}: {e}")

# -----------------------------
# Append new data, remove duplicates, and save CSV
# -----------------------------
if news_data:
    df_new = pd.DataFrame(news_data)
    
    # Combine with existing CSV
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Remove duplicates based on Stock + Headline
    df_final.drop_duplicates(subset=["Stock", "Headline"], inplace=True)
    
    # Save CSV
    df_final.to_csv(file_path, index=False)
    print(f"News headlines saved! Total records now: {len(df_final)}")
else:
    print("No new headlines to save today.")
