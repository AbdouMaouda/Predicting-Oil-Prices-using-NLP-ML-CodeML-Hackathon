# Predicting-Oil-Prices-using-NLP-ML-CodeML-Hackathon
A machine learning pipeline developed at the Code-ML Hackathon that combines time-series feature engineering and NLP sentiment analysis to forecast daily natural gas prices. Achieved 73% prediction accuracy.
---
## Features
- **Time-Series Feature Engineering** — Lag variables, rolling statistics, calendar indicators, and price deltas to capture short-term volatility and long-term energy market trends
- **NLP Sentiment Analysis** — Extracts sentiment signals from energy-related news articles using TextBlob and incorporates them as model features alongside historical price data
- **Model Comparison** — Systematic evaluation of Linear Regression and Random Forest algorithms using MAE and R2 metrics
- **Geopolitical Impact Quantification** — Measures how geopolitical events affect commodity prices through news sentiment scoring
- **Automated Submission** — Generates and exports a submission.csv with predicted prices
---
## Tech Stack
| Layer | Technology |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | Scikit-Learn |
| NLP | TextBlob |
| Environment | Google Colab |
---
## Results
| Model | MAE | R2 |
|---|---|---|
| Linear Regression (No Sentiment) | — | — |
| Linear Regression (With Sentiment) | — | — |
| Random Forest (With Sentiment) | — | 0.73 |
Random Forest with sentiment features was selected as the final model based on MAE and R2 evaluation.
---
## Project Structure
```
oil-price-prediction/
├── main.py                                        — Main ML pipeline
├── energy_price_prediction_baseline.ipynb         — Jupyter notebook
├── FinSen_US_Categorized_Timestamp.csv            — News sentiment dataset
├── train_henry_hub_natural_gas_spot_price_daily.csv — Training data
├── test-template.csv                              — Test data template
└── README.md
```
---
## How It Works
### 1. Data Loading and Cleaning
Historical natural gas prices are loaded, sorted by date, and missing values are interpolated using time-based methods.
### 2. Feature Engineering
The pipeline generates the following features:
- Lag variables — Lag 1, 7, and 30 days
- Rolling averages — 3, 7, 30, and 90 day windows
- Daily change and percentage change
- Calendar features — year, month, weekday, season
### 3. Sentiment Analysis
Energy-related news articles are filtered using keywords such as "natural gas", "oil", "pipeline", and "OPEC". TextBlob computes a daily sentiment polarity score which is merged into the training data.
### 4. Model Training
Both Linear Regression and Random Forest models are trained on an 80/20 time-based split and evaluated using MAE and R2 metrics.
### 5. Prediction
The best performing model generates predictions on the test set and exports them to submission.csv.
---
## Local Setup
### Prerequisites
- Python 3.8+
- pip
### Installation
git clone https://github.com/AbdouMaouda/oil-price-prediction.git
cd oil-price-prediction
pip install pandas numpy scikit-learn textblob
### Run
python main.py
---
## Team
Built by a 4-person team at the Code-ML Hackathon
- AbdouMaouda
---
## License
This project is licensed under the MIT License.
