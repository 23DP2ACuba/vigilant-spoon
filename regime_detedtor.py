from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from warnings import filterwarnings
import matplotlib.pyplot as plt
import hmmlearn.hmm as hmm
import yfinance as yf
import pandas as pd
import numpy as np
import pickle

'''
HMM
'''
# ------------------- Configuration -------------------
SYMBOL = 'TSLA'
folder = "TSLA"
START_DATE = '2020-01-01'
END_DATE = '2025-06-27'
PERIOD = "1d"
WINDOW_SIZE = 30
TRADING_FEES = 0.002
# -----------------------------------------------------

data = yf.Ticker(SYMBOL).history(start=START_DATE, end=END_DATE)

total_size = len(data)
train_size = int(total_size * 0.7)
gap_size = max(int(total_size * 0.1), 24)
test_size = total_size - train_size - gap_size

if test_size <= 30:
    raise ValueError("Insufficient data for meaningful backtesting")

train_data = data.iloc[:train_size].copy()
test_data = data.iloc[train_size + gap_size:].copy()

def create_features(data, train=True):
    data = data.copy()
    lookback = 6

    if len(data) < lookback + 1:
        raise ValueError(f"Need at least {lookback+1} periods for features")

    data["Return"] = data["Close"].pct_change().shift(1)
    data["Volatility"] = data["Return"].rolling(5, min_periods=1).std()
    data["Momentum"] = data["Close"].shift(1) - data["Close"].shift(6)
    data["Log_Volume"] = np.log(data["Volume"].shift(1) + 1e-6)

    return data.dropna() if train else data

try:
    train_data = create_features(train_data, train=True)
    test_data = create_features(test_data, train=False).dropna()
except ValueError as e:
    print(f"Feature creation failed: {e}")
    exit()

features = ["Return", "Volatility", "Momentum", "Log_Volume"]
scaler = StandardScaler()

try:
    X_train = scaler.fit_transform(train_data[features])
    X_test = scaler.transform(test_data[features])
except ValueError as e:
    print(f"Scaling failed: {e}")
    exit()

model = hmm.GaussianHMM(
    n_components=5,
    covariance_type="full",
    n_iter=100,
    tol=1e-4,
    verbose=True,
    random_state=42,
    init_params="stmc"
)
model.fit(X_train)
hidden_states = model.predict(X_train)
df_regimes = pd.DataFrame({
    "date": data.index[:len(hidden_states)],
    "close": data["Close"].iloc[:len(hidden_states)].values,
    "regime": hidden_states
})
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(15, 6))

for regime in np.unique(df_regimes["regime"]):
    mask = df_regimes["regime"] == regime
    ax.plot(df_regimes["date"][mask], df_regimes["close"][mask], '.', label=f"Regime {regime}")

ax.set_title("Market Regimes Detected by HMM")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
plt.show()
