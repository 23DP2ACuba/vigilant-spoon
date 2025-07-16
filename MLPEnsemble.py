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
THRESHOLD = 0.02
NUM_CLASSES = 3
NUM_MLPS = 8
HIDDEN_DIM = 64
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-2

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
    "Z-Score", "Market_State"
] + [f"Ma_t-{i}" for i in range(0, LOOKBACK + 1, 5)]

X = data[feature_cols].astype(np.float32).values
y = data["Target"].map({-1: 0, 0: 1, 1: 2}).astype(np.int64).values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
