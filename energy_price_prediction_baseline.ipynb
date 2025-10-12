
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import re
import warnings
warnings.filterwarnings("ignore")


trainFile = pd.read_csv("train_henry_hub_natural_gas_spot_price_daily 1.csv")
testFile  = pd.read_csv("test-template.csv")

trainFile = trainFile.drop(columns="source", errors="ignore")

train = trainFile.rename(columns={"date": "Date", "price_usd_per_mmbtu": "Price"})
test  = testFile.rename(columns={"id": "Date"})

train["Date"] = pd.to_datetime(train["Date"], errors="coerce")
train = train.sort_values("Date").dropna(subset=["Price"]).reset_index(drop=True)

train = train.set_index("Date")
train["Price"] = train["Price"].interpolate(method="time")
train = train.reset_index()


train["Lag_1"] = train["Price"].shift(1)
train["Lag_7"] = train["Price"].shift(7)
train["Lag_30"] = train["Price"].shift(30)

train["Rolling_mean_3"]  = train["Price"].rolling(window=3).mean()
train["Rolling_mean_7"]  = train["Price"].rolling(window=7).mean()
train["Rolling_mean_30"] = train["Price"].rolling(window=30).mean()
train["Rolling_mean_90"] = train["Price"].rolling(window=90).mean()

train["Daily_Change"]  = train["Price"] - train["Lag_1"]
train["Daily%_change"] = (train["Daily_Change"] / train["Lag_1"]) * 100

train["Year"]    = train["Date"].dt.year
train["Month"]   = train["Date"].dt.month
train["Weekday"] = train["Date"].dt.weekday

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

train["Season"] = train["Month"].apply(get_season)
train = train.dropna().reset_index(drop=True)


news = pd.read_csv("FinSen_US_Categorized_Timestamp.csv")

news["Time"] = pd.to_datetime(news["Time"], format="%d/%m/%Y", errors="coerce")
news = news[news["Time"].dt.year <= 2019].copy()

news["Text"] = news["Title"].fillna('') + " " + news["Content"].fillna('')

print(f"Loaded {len(news)} news articles (≤2019)")


keywords = [
    "gas", "natural gas", "oil", "crude", "fuel", "pipeline", "energy", "opec",
    "supply", "demand", "storage", "refinery", "price", "production", "war",
    "russia", "conflict", "economy"
]
pattern = "|".join(keywords)
energy_news = news[news["Text"].str.contains(pattern, case=False, na=False)]

def get_sentiment(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0

energy_news["Sentiment"] = energy_news["Text"].apply(get_sentiment)
daily_sentiment = energy_news.groupby("Time")["Sentiment"].mean().reset_index()
daily_sentiment.rename(columns={"Time": "Date"}, inplace=True)

print(f" Found {len(energy_news)} energy-related news articles.")


merged = pd.merge(train, daily_sentiment, on="Date", how="left")
merged["Sentiment"] = merged["Sentiment"].fillna(0)

merged["Target"] = merged["Price"].shift(-1)
merged = merged.dropna().reset_index(drop=True)

features = [
    "Lag_1", "Lag_7", "Lag_30",
    "Rolling_mean_3", "Rolling_mean_7", "Rolling_mean_30", "Rolling_mean_90",
    "Daily_Change", "Daily%_change", "Sentiment"
]

split_index = int(len(merged) * 0.8)
train_data = merged.iloc[:split_index]
test_data  = merged.iloc[split_index:]

X_train = train_data[features]
y_train = train_data["Target"]
X_test  = test_data[features]
y_test  = test_data["Target"]


print("\n Training Linear Regression Models...")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae_with_sent = mean_absolute_error(y_test, y_pred)
r2_with_sent  = r2_score(y_test, y_pred)

features_no_sent = [f for f in features if f != "Sentiment"]
model_no_sent = LinearRegression()
model_no_sent.fit(X_train[features_no_sent], y_train)
y_pred_no_sent = model_no_sent.predict(X_test[features_no_sent])

mae_no_sent = mean_absolute_error(y_test, y_pred_no_sent)
r2_no_sent  = r2_score(y_test, y_pred_no_sent)

print(f" With Sentiment → MAE: {mae_with_sent:.4f}, R²: {r2_with_sent:.4f}")
print(f" Without Sentiment → MAE: {mae_no_sent:.4f}, R²: {r2_no_sent:.4f}")


print("\n Training Random Forest Regressor...")
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, rf_pred)
r2_rf  = r2_score(y_test, rf_pred)

print(f"Random Forest → MAE: {mae_rf:.4f}, R²: {r2_rf:.4f}")

print("\n--- FINAL RESULTS ---")
print(f"Linear Regression (No Sentiment): MAE={mae_no_sent:.4f}, R²={r2_no_sent:.4f}")
print(f"Linear Regression (With Sentiment): MAE={mae_with_sent:.4f}, R²={r2_with_sent:.4f}")
print(f"Random Forest (With Sentiment): MAE={mae_rf:.4f}, R²={r2_rf:.4f}")


test["Date"] = pd.to_datetime(test["Date"], errors="coerce")

full_data = pd.concat([train, test], ignore_index=True, sort=False)

full_data["Date"] = pd.to_datetime(full_data["Date"], errors="coerce")
full_data = full_data.dropna(subset=["Date"])
full_data = full_data.sort_values("Date").reset_index(drop=True)

full_data["Lag_1"] = full_data["Price"].shift(1)
full_data["Lag_7"] = full_data["Price"].shift(7)
full_data["Lag_30"] = full_data["Price"].shift(30)
full_data["Rolling_mean_3"]  = full_data["Price"].rolling(window=3).mean()
full_data["Rolling_mean_7"]  = full_data["Price"].rolling(window=7).mean()
full_data["Rolling_mean_30"] = full_data["Price"].rolling(window=30).mean()
full_data["Rolling_mean_90"] = full_data["Price"].rolling(window=90).mean()

full_data["Daily_Change"]  = full_data["Price"] - full_data["Lag_1"]
full_data["Daily%_change"] = (full_data["Daily_Change"] / full_data["Lag_1"]) * 100

full_data["Year"] = full_data["Date"].dt.year
full_data["Month"] = full_data["Date"].dt.month
full_data["Weekday"] = full_data["Date"].dt.weekday
full_data["Season"] = full_data["Month"].apply(get_season)

full_merged = pd.merge(full_data, daily_sentiment, on="Date", how="left")
full_merged["Sentiment"] = full_merged["Sentiment"].fillna(0)

full_merged = full_merged.fillna(method="ffill")
full_merged["Sentiment"] = full_merged["Sentiment"].replace(0, full_merged["Sentiment"].median())

test_ready = full_merged[full_merged["Date"].isin(test["Date"])].copy()

X_final_test = test_ready[features].fillna(method="ffill").fillna(method="bfill")

test_predictions = rf.predict(X_final_test)

submission = pd.DataFrame({
    "Date": test_ready["Date"],
    "Predicted_Price": test_predictions
})

submission.to_csv("submission.csv", index=False)

print(" Predictions successfully saved to submission.csv")

print("\n📄 Submission file preview:")
display(submission.head())



from google.colab import files
files.download("submission.csv")


