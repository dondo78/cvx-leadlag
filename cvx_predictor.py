# Script Python automatico per previsione giornaliera CVX con invio Telegram

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE TELEGRAM ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")

def invia_telegram(messaggio):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': messaggio}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Errore invio Telegram: {e}")

# --- 1. Scarica dati ---
period = "12mo"
xom = yf.download("XOM", period=period, auto_adjust=True)
cvx = yf.download("CVX", period=period, auto_adjust=True)
cvx_raw = yf.download("CVX", period=period, interval="1d", auto_adjust=False)

# --- Verifica che i dati siano validi ---
if xom.empty or cvx.empty or cvx_raw.empty:
    raise ValueError("Uno o più dataset risultano vuoti. Controlla i simboli o la connessione internet.")

# --- 2. Calcola ritorni e feature ---
data = pd.concat([
    xom['Close'].rename('XOM_Close'),
    cvx['Close'].rename('CVX_Close')
], axis=1)

data['XOM_Return'] = data['XOM_Close'].pct_change()
data['CVX_Return'] = data['CVX_Close'].pct_change()
data['Spread'] = data['XOM_Return'] - data['CVX_Return']
data['Zscore'] = (data['Spread'] - data['Spread'].rolling(10).mean()) / data['Spread'].rolling(10).std()
data['CVX_Volatility'] = data['CVX_Return'].rolling(5).std()
data['Corr'] = data['XOM_Return'].rolling(5).corr(data['CVX_Return'])

# Feature laggate
for lag in range(1, 4):
    data[f'XOM_Return_lag{lag}'] = data['XOM_Return'].shift(lag)
    data[f'CVX_Return_lag{lag}'] = data['CVX_Return'].shift(lag)

# --- 3. Target: se CVX sale nella giornata ---
cvx_raw = cvx_raw[['Open', 'Close']]
cvx_raw['Target'] = (cvx_raw['Close'] > cvx_raw['Open']).astype(int)

# Allinea le date con il DataFrame principale
cvx_raw = cvx_raw.loc[data.index.intersection(cvx_raw.index)]
data = data.loc[cvx_raw.index]
data['Target'] = cvx_raw['Target']

# --- 4. Prepara dati per il modello ---
data = data.dropna()
features = [col for col in data.columns if col not in ['XOM_Close', 'CVX_Close', 'Target']]
X = data[features]
y = data['Target']

# --- 5. Addestra il modello ---
X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 6. Previsione per oggi ---
X_today = X.tail(1)
pred = model.predict(X_today)[0]
prob = model.predict_proba(X_today)[0][pred] * 100

emoji = "📈 Salirà" if pred == 1 else "📉 Scenderà"
messaggio = f"📊 Previsione giornaliera CVX:\nRisultato: {emoji}\nConfidenza: {prob:.1f}%"
invia_telegram(messaggio)

# --- 7. Report (facoltativo) ---
print("Report modello:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
