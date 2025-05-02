# iclstm_train_pytorch_optimized.py
"""End‑to‑end training script for an Interpretable Convex LSTM (IC‑LSTM)
with properly separated training / validation / test splits, correct
scaling, and more robust engineering practices.
"""
from __future__ import annotations

import os
import time
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# -----------------------------------------------------------------------------
# Reproducibility ----------------------------------------------------------------
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Hyper‑parameters -------------------------------------------------------------
# -----------------------------------------------------------------------------
SEQ_LEN = 10
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 1e-3
HIDDEN_UNITS = 128
VAL_RATIO = 0.15    # proportion of training set held out for validation

# -----------------------------------------------------------------------------
# Paths -----------------------------------------------------------------------
# -----------------------------------------------------------------------------
VISUAL_DIR = Path("visualization/iclstm_train_pytorch_optimized")
VISUAL_DIR.mkdir(parents=True, exist_ok=True)
TEST_RESULT_PATH = VISUAL_DIR / "test_result.txt"
MODEL_PATH = VISUAL_DIR / "iclstm_model.pt"
PLOT_PATH = VISUAL_DIR / "ICLSTM_loss.png"
SCALER_STATS_PATH = VISUAL_DIR / "scaler_stats.npz"

# -----------------------------------------------------------------------------
# Columns ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
INPUT_COLS = [
    "Bd_T_HP_supply", "Bd_T_HP_return",
    "Z01_T", "Z02_T", "Z03_T", "Z04_T", "Z05_T", "Z06_T", "Z07_T", "Z08_T",
    "Bd_FracCh_Bat", "Fa_ECh_Bat", "Fa_EDCh_Bat", "Fa_Pw_Prod",
    "PV_Gen_corrected", "Fa_E_All",
    "P1_T_Thermostat_sp_out", "P2_T_Thermostat_sp_out",
    "P3_T_Thermostat_sp_out", "P4_T_Thermostat_sp_out",
    "Bd_Pw_Bat_sp_out", "Bd_T_HP_sp_out",
]

TARGET_COLS = [
    "Bd_T_HP_supply", "Bd_T_HP_return",
    "Z01_T", "Z02_T", "Z03_T", "Z04_T", "Z05_T", "Z06_T", "Z07_T", "Z08_T",
    "Bd_FracCh_Bat", "Fa_ECh_Bat", "Fa_EDCh_Bat", "Fa_Pw_Prod",
    "PV_Gen_corrected", "Fa_E_All",
]

# -----------------------------------------------------------------------------
# Data utilities --------------------------------------------------------------
# -----------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """Dataset of (sequence, target) tuples."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:  # noqa: D401
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a dataframe into (X, y) sequence arrays."""
    data = df[INPUT_COLS + TARGET_COLS].values.astype(np.float32)

    X, Y = [], []
    for i in range(len(data) - SEQ_LEN):
        X.append(data[i : i + SEQ_LEN, : len(INPUT_COLS)])
        Y.append(data[i + SEQ_LEN, len(INPUT_COLS) :])

    X = np.stack(X)
    Y = np.stack(Y)

    # Optionally augment with negative features for IC‑LSTM as in the paper.
    X = np.concatenate([X, -X], axis=2)
    return X, Y


# -----------------------------------------------------------------------------
# IC‑LSTM implementation -------------------------------------------------------
# -----------------------------------------------------------------------------
class ICLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Following original IC‑LSTM paper initialisation.
        self.Wi = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.Ui = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.DWi = nn.Parameter(torch.rand(hidden_size))
        self.bi = nn.Parameter(torch.zeros(hidden_size))

        self.DWf = nn.Parameter(torch.rand(hidden_size))
        self.bf = nn.Parameter(torch.zeros(hidden_size))

        self.DWo = nn.Parameter(torch.rand(hidden_size))
        self.bo = nn.Parameter(torch.zeros(hidden_size))

        self.DWc = nn.Parameter(torch.rand(hidden_size))
        self.bc = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, state):
        h_tm1, c_tm1 = state
        xi = F.linear(x, self.Wi)
        ui = F.linear(h_tm1, self.Ui)

        i = F.relu(self.DWi * (xi + ui) + self.bi)
        f = F.relu(self.DWf * (xi + ui) + self.bf)
        o = F.relu(self.DWo * (xi + ui) + self.bo)
        c_bar = F.relu(self.DWc * (xi + ui) + self.bc)

        c_t = f * c_tm1 + i * c_bar
        h_t = o * F.relu(c_t)
        return h_t, c_t

    def clamp_nonneg(self):
        for p in self.parameters():
            p.data.clamp_(min=0.0)


class ICLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.cell1 = ICLSTMCell(input_size, hidden_size)
        self.cell2 = ICLSTMCell(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.fc_out = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        h1 = x.new_zeros(B, self.cell1.hidden_size)
        c1 = x.new_zeros(B, self.cell1.hidden_size)
        outputs1 = []
        for t in range(T):
            h1, c1 = self.cell1(x[:, t], (h1, c1))
            outputs1.append(h1.unsqueeze(1))
        seq1 = torch.cat(outputs1, dim=1)
        res1 = F.relu(self.fc1(seq1)) + x

        h2 = x.new_zeros(B, self.cell2.hidden_size)
        c2 = x.new_zeros(B, self.cell2.hidden_size)
        outputs2 = []
        for t in range(T):
            h2, c2 = self.cell2(res1[:, t], (h2, c2))
            outputs2.append(h2.unsqueeze(1))
        seq2 = torch.cat(outputs2, dim=1)
        res2 = F.relu(self.fc2(seq2)) + x

        out_seq = self.fc_out(res2)
        return out_seq[:, -1]

    def clamp_weights(self):
        self.cell1.clamp_nonneg()
        self.cell2.clamp_nonneg()
        for layer in (self.fc1, self.fc2, self.fc_out):
            layer.weight.data.clamp_(min=0.0)


# -----------------------------------------------------------------------------
# Train / validate / test loops ----------------------------------------------
# -----------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimiser):
    model.train()
    running_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimiser.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimiser.step()
        model.clamp_weights()
        running_loss += loss.item() * Xb.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            loss += criterion(preds, yb).item() * Xb.size(0)
    return loss / len(loader.dataset)


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    # --------------------------- Load & preprocess ---------------------------
    df_raw = pd.read_csv("simulation_output.csv")
    X, y = build_sequences(df_raw)

    # --- Train / val / test split (no shuffling to preserve chronology) ------
    n_total = len(X)
    n_test = int(0.3 * n_total)
    n_trainval = n_total - n_test
    n_val = int(VAL_RATIO * n_trainval)
    n_train = n_trainval - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[-n_test:], y[-n_test:]

    # --- Scaling -------------------------------------------------------------
    x_scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[2]))
    y_scaler = StandardScaler().fit(y_train)

    def scale_X(x: np.ndarray) -> np.ndarray:
        return x_scaler.transform(x.reshape(-1, x.shape[2])).reshape(x.shape)

    X_train = scale_X(X_train)
    X_val = scale_X(X_val)
    X_test = scale_X(X_test)

    y_train = y_scaler.transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    # Save scaler statistics for inference.
    np.savez(
        SCALER_STATS_PATH,
        input_mean=x_scaler.mean_,
        input_std=x_scaler.scale_,
        output_mean=y_scaler.mean_,
        output_std=y_scaler.scale_,
    )

    # --------------------------- DataLoaders ---------------------------------
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # --------------------------- Model & optim --------------------------------
    model = ICLSTM(input_size=X_train.shape[2], hidden_size=HIDDEN_UNITS, output_size=y_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimiser)
        val_loss = evaluate(model, val_loader, criterion)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} — train: {train_loss:.5f} — val: {val_loss:.5f}")

    # --------------------------- Save model ----------------------------------
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved ➜ {MODEL_PATH}")

    # --------------------------- Test evaluation -----------------------------
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds_list.append(model(Xb.to(device)).cpu().numpy())
            targets_list.append(yb.numpy())

    y_pred_scaled = np.vstack(preds_list)
    y_true_scaled = np.vstack(targets_list)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_true_scaled)

    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    mape = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    with open(TEST_RESULT_PATH, "w") as f:
        f.write("Test results\n")
        f.write("============\n")
        f.write(f"MSE per target : {mse}\n")
        f.write(f"MAPE per target: {mape}\n")
        f.write(f"R2 per target  : {r2}\n\n")
        f.write(f"MSE (avg)  : {mse.mean():.4f}\n")
        f.write(f"MAPE (avg) : {mape.mean():.4f}\n")
        f.write(f"R2 (avg)   : {r2.mean():.4f}\n\n")
        f.write(f"Train samples  : {len(train_loader.dataset)}\n")
        f.write(f"Val samples    : {len(val_loader.dataset)}\n")
        f.write(f"Test samples   : {len(test_loader.dataset)}\n")
        f.write(f"Input features : {X_train.shape[2]}\n")
        f.write(f"Sequence length: {SEQ_LEN}\n")
        f.write(f"Targets        : {y_train.shape[1]}\n")
        f.write(f"Batch size     : {BATCH_SIZE}\n")
        f.write(f"Epochs         : {EPOCHS}\n")
        f.write(f"Training time  : {time.time() - start_time:.2f} s\n")

    # --------------------------- Plot losses ---------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="Train")
    plt.plot(history["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Loss plot saved ➜ {PLOT_PATH}")

    # --------------------------- Per‑target plots ----------------------------
    for i, col in enumerate(TARGET_COLS):
        plt.figure()
        plt.plot(y_true[:, i], label="True")
        plt.plot(y_pred[:, i], label="Pred")
        plt.title(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(VISUAL_DIR / f"ICLSTM_compare_{col}.png")

    print("All done! 😊")
