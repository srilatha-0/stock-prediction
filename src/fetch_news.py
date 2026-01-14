import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

stocks = ["RELIANCE", "TCS", "INFY"]
data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)

news_data = []

for stock in stocks:
    url = f"https://news.google.com/search?q={stock}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    headlines = [h.text for h in soup.find_all("a")][:5]  # first 5 headlines
    for headline in headlines:
        news_data.append({"Stock": stock, "Headline": headline})

df_news = pd.DataFrame(news_data)
df_news.to_csv(f"{data_dir}/news_headlines.csv", index=False)
print("News headlines saved!")
