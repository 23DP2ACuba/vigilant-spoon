import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

'''
MLP Ensemble
'''

# ------------------- Configuration -------------------
SYMBOL = 'TSLA'
folder = "TSLA"
START_DATE = '2020-01-01'
END_DATE = '2025-06-27'
PERIOD = "1d"
WINDOW_SIZE = 20
TRADING_FEES = 0.002
LOOKBACK = 20
MA_PERIOD = 20
# -----------------------------------------------------

data = yf.Ticker(SYMBOL).history(start=START_DATE, end=END_DATE)
data = data[["Open", "High", "Low", "Close", "Volume"]]

def create_features(data):
    """Create features with proper lagging"""
    df = data.copy()
    df['return'] = df['Close'].pct_change().shift(1)
    df['volatility'] = df['return'].rolling(5, min_periods=1).std().shift(1)
    df['momentum'] = (df['Close'].shift(1) - df['Close'].shift(6))
    df['log_volume'] = np.log(df['Volume'].shift(1) + 1e-6)
    df["ma"] = df.Close.rolling(MA_PERIOD).mean()
    df['cl_to_ma_pct'] = (data.Close - df.ma) / data.Close * 100
    mean = df['return'].rolling(WINDOW_SIZE).mean()
    std = df['return'].rolling(WINDOW_SIZE).std()
    df["z-score"] = (df['return'] - mean) / std
    for i in range(0, LOOKBACK + 1, 5):
      df[f"ma_t-{i}"] = df["ma"].shift(i)

    df["target"] = 
    return df.dropna()

data = create_features(data)
data.head(5)
