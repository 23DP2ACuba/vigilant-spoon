import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
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
START_DATE = '2018-01-01'
END_DATE = '2025-06-27'
PERIOD = "1d"
WINDOW_SIZE = 30
TRADING_FEES = 0.004
LOOKBACK = 20
MA_PERIOD = 20
N_DAYS = 5
THRESHOLD = 0.03
NUM_CLASSES = 3
NUM_MLPS = 4
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-2
LABEL_SMOOTHING = 0.9

hmm_features = ["Return", "Volatility", "Momentum", "Log_Volume"]
# -----------------------------------------------------

# ------------------- Data Prep -------------------
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
    df['Volatility'] = df['Return'].rolling(5).std().shift(1)
    df['Momentum'] = df['Close'].shift(1) - df['Close'].shift(6)
    df['Log_Volume'] = np.log(df['Volume'].shift(1) + 1e-6)
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
print(data["Target"].value_counts())
# ------------------- Dataset -------------------
feature_cols = [
    "Return", "Volatility", "Momentum", "Log_Volume", "Cl_to_Ma_pct",
    "Z-Score", "Market_State"
] + [f"Ma_t-{i}" for i in range(0, LOOKBACK + 1, 5)]

X = data[feature_cols].astype(np.float32).values
y = data["Target"].map({-1: 0, 0: 1, 1: 2}).astype(np.int64).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

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

# ------------------- MLP & Ensemble -------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MPLEnsemble(nn.Module):
    def __init__(self, num_mlps, input_dim, hidden_dim, output_dim, dropout=0.3, use_lstm=True):
        super().__init__()
        self.num_mlps = num_mlps
        self.use_lstm = use_lstm

        self.ensemble = nn.ModuleList([
            MLP(input_dim, hidden_dim, output_dim, dropout) for _ in range(num_mlps)
        ])

        self.softmax = nn.Softmax(dim=-1)

        if use_lstm:
            self.lstm = nn.LSTM(output_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        else:
            self.proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(output_dim * num_mlps, output_dim)
            )

    def forward(self, x):
        logits = [mlp(x) for mlp in self.ensemble]
        softmaxed = [self.softmax(logit) for logit in logits]

        if self.use_lstm:
            stacked = torch.stack(softmaxed, dim=1)
            lstm_out, _ = self.lstm(stacked)
            final = self.fc(lstm_out[:, -1, :])
            return final
        else:
            combined = torch.cat(logits, dim=-1)
            out = self.proj(combined)
            return out

    def get_confidence(self, x):
        """Optional: return average confidence for inspection"""
        softmaxed = [self.softmax(mlp(x)) for mlp in self.ensemble]
        stacked = torch.stack(softmaxed, dim=0)
        return stacked.mean(dim=0)
# ------------------- Training -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MPLEnsemble(NUM_MLPS, X.shape[1], HIDDEN_DIM, NUM_CLASSES, dropout=DROPOUT).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=LABEL_SMOOTHING, weight=weights)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# ------------------- Evaluation -------------------
model.eval()
all_preds, all_labels, all_confidences = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        
        logits = model(xb)
        confidences = model.get_confidence(xb)
        preds = torch.argmax(confidences, dim=1).cpu().numpy()
        max_conf = torch.max(confidences, dim=1).values.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(yb.numpy())
        all_confidences.extend(max_conf)

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds, target_names=["Sell", "Hold", "Buy"]))

import matplotlib.pyplot as plt
plt.hist(all_confidences, bins=20)
plt.title("Prediction Confidence Histogram")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.show()

print("Average Confidence: {:.4f}".format(np.mean(all_confidences)))


