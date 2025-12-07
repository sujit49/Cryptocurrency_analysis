# model_utils.py
import numpy as np
import pandas as pd

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_indicators(df):
    # ensure Close exists
    df = df.copy()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_14'] = df['Close'].rolling(window=14).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].pct_change().rolling(window=14).std()
    df['RSI'] = compute_rsi(df['Close'], period=14)
    # keep the columns in a stable order so scaler and inverse transform are consistent
    # we'll use: Close, EMA_20, MA_7, MA_14, Returns, Volatility, RSI
    df = df[['Close','EMA_20','MA_7','MA_14','Returns','Volatility','RSI','Volume']]
    return df

def create_sequences(data, seq_len=60):
    """
    data: numpy array (n_rows, n_features)
    returns X (n_samples, seq_len, n_features), y (n_samples, 1) where y is next-step first column (Close)
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # predict Close
    return np.array(X), np.array(y).reshape(-1,1)
