import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import os

# ------------------- Configuration -------------------
SYMBOL = 'TSLA'
folder = "TSLA"
START_DATE = '2020-01-01'
END_DATE = '2025-06-27'
PERIOD = "1d"
WINDOW_SIZE = 20
TRADING_FEES = 0.002
LOOKBACK = 8
MA_PERIOD = 20
N_DAYS = 5
THRESHOLD = 0.03
NUM_CLASSES = 3
NUM_MLPS = 64
HIDDEN_DIM = 64
DROPOUT = 0.4
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-2
RSI_WINDOW = 14
LABEL_SMOOTHING = 0.1
hmm_features = ["Return", "Volatility", "Momentum", "Log_Volume"]
# -----------------------------------------------------

# Load data
data = yf.Ticker(SYMBOL).history(start=START_DATE, end=END_DATE)
data = data[["Open", "High", "Low", "Close", "Volume"]]

def add_hmm(df):
    with open(os.path.join(folder, "hmm_model.pkl"), "rb") as f:
        hmm_model = pickle.load(f)

    with open(os.path.join(folder, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X = df[hmm_features]
    X_scaled = scaler.transform(X)
    return hmm_model.predict(X_scaled)

def create_features(data):
    df = data.copy()
    df['Return'] = df['Close'].pct_change().shift(1)
    df['Volatility'] = df['Return'].rolling(5, min_periods=1).std().shift(1)
    df['Momentum'] = df['Close'].shift(1) - df['Close'].shift(6)
    df['Log_Volume'] = np.log(df['Volume'].shift(1))
    df["Ma"] = df['Close'].rolling(MA_PERIOD).mean()
    df['Cl_to_Ma_pct'] = (df['Close'] - df['Ma']) / df['Close'] * 100
    df["Z-Score"] = (df['Return'] - df['Return'].rolling(WINDOW_SIZE).mean()) / df['Return'].rolling(WINDOW_SIZE).std()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=RSI_WINDOW).mean()
    loss = -delta.clip(upper=0).rolling(window=RSI_WINDOW).mean()
    rs = gain / (loss + 1e-6)
    df['RSI'] = 100 - (100 / (1 + rs))

    for i in range(0, LOOKBACK + 1, 5):
        df[f"Ma_t-{i}"] = df["Ma"].shift(i)

    df['Future_Return'] = (df['Close'].shift(-N_DAYS) - df['Close']) / df['Close']
    df["Target"] = 0
    df.loc[df['Future_Return'] > THRESHOLD, "Target"] = 1
    df.loc[df['Future_Return'] < -THRESHOLD, "Target"] = -1

    df = df.dropna()
    df["Market_State"] = add_hmm(df)

    return df
data = create_features(data)

feature_cols = [
    "Return", "Volatility", "Momentum", "Log_Volume", "Cl_to_Ma_pct",
    "Z-Score", "Market_State", "RSI"
] + [f"Ma_t-{i}" for i in range(0, LOOKBACK + 1, 5)]

X = data[feature_cols].astype(np.float32).values
y = data["Target"].map({-1: 0, 0: 1, 1: 2}).astype(np.int64).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_components = 10
pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ------------------ Model Definition ------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class MPLEnsemble(nn.Module):
    def __init__(self, num_mlps, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ensemble = nn.ModuleList([
            MLP(input_dim, hidden_dim, output_dim) for _ in range(num_mlps)
        ])
        self.proj = nn.Linear(output_dim * num_mlps, NUM_CLASSES)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        outputs = [mlp(x) for mlp in self.ensemble]
        out = torch.cat(outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out
# ------------------- Label Smoothing Loss -------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, weight=None):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.weight = weight
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        if self.weight is not None:
            weighted = self.weight[target].unsqueeze(1)
            return (self.kl(pred, true_dist) * weighted).mean()
        return self.kl(pred, true_dist)

# ------------------ Training Loop ------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MPLEnsemble(NUM_MLPS, X.shape[1], HIDDEN_DIM, NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ------------------ Evaluation ------------------

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, target_names=["Sell", "Hold", "Buy"]))

label_map = {0: -1, 1: 0, 2: 1}
pred_labels = np.array([label_map[p] for p in all_preds])

test_index = data.iloc[len(data) - len(X_test):].index
returns = data.loc[test_index, 'Future_Return'].values

strategy_returns = pred_labels * returns
strategy_returns = strategy_returns - np.abs(pred_labels) * TRADING_FEES  

cumulative_returns = (1 + strategy_returns).cumprod()

print(f"\n📈 Final Cumulative Return: {cumulative_returns[-1]:.4f} ({(cumulative_returns[-1] - 1) * 100:.2f}%)")

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.plot(cumulative_returns, label="Strategy")
plt.axhline(1.0, color="gray", linestyle="--", label="Break-even")
plt.title("Cumulative Returns of Trading Strategy")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
