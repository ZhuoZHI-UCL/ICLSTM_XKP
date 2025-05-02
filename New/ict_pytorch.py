import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# 1. 设置随机种子和路径
# -------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
visualization_path = 'visualization/ict_pytorch/'
os.makedirs(visualization_path, exist_ok=True)
test_result_path = os.path.join(visualization_path, 'test_result.txt')
model_path = os.path.join(visualization_path, 'ict_pytorch.pth')

# -------------------------------------------------
# 2. 数据读取与基本配置
# -------------------------------------------------
sequence_length = 10
epochs = 2000
batch_size = 256

input_cols = ['Bd_T_HP_supply', 'Bd_T_HP_return', 'Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 'Z06_T', 'Z07_T', 'Z08_T',
              'Bd_FracCh_Bat', 'Fa_ECh_Bat', 'Fa_EDCh_Bat', 'Fa_Pw_Prod', 'PV_Gen_corrected', 'Fa_E_All',
              'P1_T_Thermostat_sp_out', 'P2_T_Thermostat_sp_out', 'P3_T_Thermostat_sp_out', 'P4_T_Thermostat_sp_out',
              'Bd_Pw_Bat_sp_out', 'Bd_T_HP_sp_out']
predict_cols = ['Bd_T_HP_supply', 'Bd_T_HP_return', 'Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 'Z06_T', 'Z07_T', 'Z08_T',
                'Bd_FracCh_Bat', 'Fa_ECh_Bat', 'Fa_EDCh_Bat', 'Fa_Pw_Prod', 'PV_Gen_corrected', 'Fa_E_All']

data = pd.read_csv('./New/simulation_output_whole_year.csv')
# Select columns present
input_columns = [c for c in input_cols if c in data.columns]
target_columns = [c for c in predict_cols if c in data.columns]
input_data = data[input_columns].values
target_data = data[target_columns].values

# Build sequences
X, Y = [], []
for i in range(len(data) - sequence_length):
    X.append(input_data[i: i + sequence_length])
    Y.append(target_data[i + sequence_length])
X = np.stack(X)
Y = np.stack(Y)
# Symmetric concatenation
X = np.concatenate([X, -X], axis=2)

# Train-test split
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=123, shuffle=False
)
# Further split train into train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.25, random_state=123, shuffle=False
)

num_steps, num_dims = X_train.shape[1], X_train.shape[2]
num_targets = len(target_columns)
print(f'Using {num_dims} dims, {num_steps} steps, {num_targets} targets')

# Standardization
scaler_X = preprocessing.StandardScaler().fit(X_train.reshape(-1, num_dims))
scaler_y = preprocessing.StandardScaler().fit(y_train)
X_train = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1, num_steps, num_dims)
X_val = scaler_X.transform(X_val.reshape(-1, num_dims)).reshape(-1, num_steps, num_dims)
X_test = scaler_X.transform(X_test.reshape(-1, num_dims)).reshape(-1, num_steps, num_dims)
y_train_scaled = scaler_y.transform(y_train)
y_val_scaled = scaler_y.transform(y_val)

print('Input mean:', scaler_X.mean_)
print('Input std:', scaler_X.scale_)
print('Output mean:', scaler_y.mean_)
print('Output std:', scaler_y.scale_)

# -------------------------------------------------
# Dataset and DataLoader
# -------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

train_dataset = TimeSeriesDataset(X_train, y_train_scaled)
val_dataset   = TimeSeriesDataset(X_val,   y_val_scaled)
test_dataset  = TimeSeriesDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# -------------------------------------------------
# Custom Modules
# -------------------------------------------------
class ConvexMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, r=0.3, tau=1.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.r = r
        self.tau = tau
        self.w_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.xavier_uniform_(self.w_proj.weight)
        self.d_q = nn.Parameter(torch.ones(d_model))
        self.d_k = nn.Parameter(torch.ones(d_model))
        self.d_v = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        B, T, D = x.shape
        Xp = torch.relu(self.w_proj(x))
        Q = Xp * self.d_q
        K = Xp * self.d_k
        V = Xp * self.d_v
        def split(x): return x.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        Qh, Kh, Vh = split(Q), split(K), split(V)
        scores = torch.matmul(Qh, Kh.transpose(-2,-1))
        z = (scores - self.r) / max(self.tau, 1e-9)
        z_max,_ = z.max(dim=-1, keepdim=True)
        exp_z = torch.exp(z - z_max)
        attn = exp_z / (exp_z.sum(dim=-1, keepdim=True) + 1e-9)
        out = torch.matmul(attn, Vh)
        out = out.transpose(1,2).contiguous().view(B, T, D)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, r=0.3, tau=1.0):
        super().__init__()
        self.att = ConvexMultiHeadAttention(d_model, num_heads, r, tau)
        self.dropout1 = nn.Dropout(rate)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.dropout2 = nn.Dropout(rate)
    def forward(self, x):
        att = self.att(x)
        x = x + self.dropout1(att)
        ffn = self.ffn(x)
        out = x + self.dropout2(ffn)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.token_proj = nn.Linear(num_dims, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
    def forward(self, x):
        B, T, _ = x.size()
        x = self.token_proj(x)
        positions = torch.arange(T, device=x.device)
        return x + self.pos_emb(positions).unsqueeze(0)

class TransformerModel(nn.Module):
    def __init__(self, seq_len, num_dims, num_targets,
                 d_model=64, num_heads=1, ff_dim=64, num_layers=1,
                 rate=0.1, r=1.0, tau=10.0):
        super().__init__()
        self.pos_emb = PositionalEmbedding(seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, rate, r, tau)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, num_targets)
    def forward(self, x):
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]
        return self.head(x)

# -------------------------------------------------
# Instantiate Model, Loss, Optimizer
# -------------------------------------------------
model = TransformerModel(
    seq_len=num_steps, num_dims=num_dims, num_targets=num_targets,
    d_model=64, num_heads=1, ff_dim=64, num_layers=1,
    rate=0.1, r=1.0, tau=10.0
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------------------------------------
# Training Loop with Early Stopping and Val Loss
# -------------------------------------------------
best_val_loss = float('inf')
patience = 5
wait = 0
train_losses, val_losses = [], []
start_time = time.time()
for epoch in range(1, epochs+1):
    # Train
    model.train()
    epoch_train_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * Xb.size(0)
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    # Validate
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, yb)
            epoch_val_loss += loss.item() * Xb.size(0)
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    # Early Stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), model_path)
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    # Print losses
    print(f'Epoch {epoch}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}')
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f}s, Epochs: {epoch}")

# Load best model
model.load_state_dict(torch.load(model_path))

# -------------------------------------------------
# Test Evaluation
# -------------------------------------------------
model.eval()
all_preds = []
with torch.no_grad():
    for Xb in test_loader:
        Xb = Xb.to(device)
        preds = model(Xb).cpu().numpy()
        all_preds.append(preds)
# ... (rest unchanged)

y_pred_scaled = np.vstack(all_preds)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test, y_pred_real, multioutput='raw_values')
mape = mean_absolute_percentage_error(y_test, y_pred_real, multioutput='raw_values')
r2 = r2_score(y_test, y_pred_real, multioutput='raw_values')

print('Test MSE:', mse)
print('Test MAPE:', mape)
print('Test R2:', r2)

# Save test results
with open(test_result_path, 'w') as f:
    f.write(f"Test MSE: {mse}\nMSE Avg: {mse.mean()}\n")
    f.write(f"Test MAPE: {mape}\nMAPE Avg: {mape.mean()}\n")
    f.write(f"Test R2: {r2}\nR2 Avg: {r2.mean()}\n")
    f.write(f"Train Samples: {len(train_dataset)}\nTest Samples: {len(test_dataset)}\n")
    f.write(f"Trauin Time: {end_time - start_time:.2f}s\n")
    f.write(f"Epochs: {epoch}\n")

# -------------------------------------------------
# 8. Plot Loss and Predictions
# -------------------------------------------------
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(visualization_path, 'Transformer_custom_loss.png'))

for i, col in enumerate(target_columns):
    plt.figure()
    plt.plot(y_test[:, i], label=f'True {col}')
    plt.plot(y_pred_real[:, i], label=f'Pred {col}')
    plt.title(f'Comparison: {col}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(visualization_path, f'Transformer_comparison_{col}.png'))
