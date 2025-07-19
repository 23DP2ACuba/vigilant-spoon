"""
BERT classification
"""
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

device =  "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_LEN = 21
N_SEGMENTS = 3
EMBEDDING_DIM = 768
DROPOUT = 0.3
ATTN_HEADS = 4
N_LAYERS = 12
LR = 2e-5
EPOCHS = 100

def add_mlp_outputs(df, mlpensemble, scaler, pca):
    mlpensemble.eval()
    features = df[feature_cols].astype(np.float32).values
    features = scaler.transform(features)
    features = pca.transform(features)

    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32).to(device)
        _, softmax_output = mlpensemble(inputs) 
        softmax_output = softmax_output.cpu().numpy()

    df["MLP_Prob_Sell"] = softmax_output[:, 0]
    df["MLP_Prob_Hold"] = softmax_output[:, 1]
    df["MLP_Prob_Buy"] = softmax_output[:, 2]

    return df


class BERTEmbedding(nn.Module):
  def __init__(self, feature_dim, max_len, embed_dim, dropout):
    super().__init__()
    self.projection = nn.Linear(feature_dim, embed_dim)
    self.pos_emb = nn.Embedding(max_len, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x_len = x.size(1)
    pos_ids = torch.arange(x_len, device = x.device).unsqueeze(0).expand(x.size(0), x_len)
    embeddings =  self.projection(x) + self.pos_emb(pos_ids)
    return self.dropout(embeddings)

class BERT(nn.Module):
  def __init__(self, feature_dim, max_len, embd_dim, n_layers, attn_heads, dropout):
    super().__init__()
    self.embedding = BERTEmbedding(feature_dim, max_len, embd_dim, dropout)
    self.encoder_layer = nn.TransformerEncoderLayer(embd_dim, attn_heads, embd_dim*4)
    self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)
    self.cls_head = nn.Linear(embd_dim, 2)

  def forward(self, x, attention_mask=None):
    x = self.embedding(x)
    x = x.permute(1, 0, 2)
    if attention_mask is not None:
      attention_mask = attention_mask == 0
    x = self.encoder_block(x, src_key_padding_mask=attention_mask)
    cls_token = x[0]
    return self.cls_head(cls_token)


feature_cols = [
    "Return", "Volatility", "Momentum", "Log_Volume", "Cl_to_Ma_pct",
    "Z-Score", "Market_State", "RSI"
] + [f"Ma_t-{i}" for i in range(0, LOOKBACK + 1, 5)]

X = data[feature_cols].astype(np.float32).values
y = data["Target"].map({-1: 0, 0: 1, 1: 2}).astype(np.int64).values

data = add_mlp_outputs(data, mlpensemble, scaler, pca)

train_dataset, val_dataset = train_test_split(data, test_size=0.2, stratify=data["Target"], random_state=42)

DATA_LEN = data.shape[1]
print(DATA_LEN)


def create_sequences(data, feature_cols, window_size):
    X_seq, y_seq = [], []
    for i in range(len(data) - window_size):
        X_seq.append(data[feature_cols].iloc[i:i + window_size].values)
        y_seq.append(data["Target"].iloc[i + window_size])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(data, feature_cols, window_size=20)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.long)
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

feature_dim = len(feature_cols)
bert = BERT(feature_dim, MAX_LEN, EMBEDDING_DIM, N_LAYERS, ATTN_HEADS, DROPOUT).to(device)
print(data[["MLP_Prob_Sell", "MLP_Prob_Hold", "MLP_Prob_Buy"]])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(bert.parameters(), lr=LR)

bert.train()
for epoch in range(EPOCHS):
  total_loss = 0

  for batch in train_loader:
    input_seq = batch[0].to(device)
    labels = batch[1].to(device)

    attention_mask = (input_seq != 0).any(-1).int()  
    y_pred = bert(input_seq, attention_mask)
    loss = criterion(y_pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()


  print(f"Epoch: {epoch}, Loss: {total_loss}")
