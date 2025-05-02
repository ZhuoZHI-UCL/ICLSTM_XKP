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
visualization_path = 'visualization/t_pytorch/'
os.makedirs(visualization_path, exist_ok=True)
test_result_path = os.path.join(visualization_path, 'test_result.txt')
model_path = os.path.join(visualization_path, 't_pytorch.pth')

# -------------------------------------------------
# 2. 数据读取与基本配置
# -------------------------------------------------
sequence_length = 5
epochs = 2000
batch_size = 256

input_cols = ['Bd_T_HP_supply', 'Bd_T_HP_return', 'Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 'Z06_T', 'Z07_T', 'Z08_T',
              'Bd_FracCh_Bat', 'Fa_ECh_Bat', 'Fa_EDCh_Bat', 'Fa_Pw_Prod', 'PV_Gen_corrected', 'Fa_E_All',
              'P1_T_Thermostat_sp_out', 'P2_T_Thermostat_sp_out', 'P3_T_Thermostat_sp_out', 'P4_T_Thermostat_sp_out',
              'Bd_Pw_Bat_sp_out', 'Bd_T_HP_sp_out']
predict_cols = ['Bd_T_HP_supply', 'Bd_T_HP_return', 'Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 'Z06_T', 'Z07_T', 'Z08_T',
                'Bd_FracCh_Bat', 'Fa_ECh_Bat', 'Fa_EDCh_Bat', 'Fa_Pw_Prod', 'PV_Gen_corrected', 'Fa_E_All']

data = pd.read_csv('./New/simulation_output_whole_year.csv')
input_columns = [c for c in input_cols if c in data.columns]
target_columns = [c for c in predict_cols if c in data.columns]
input_data = data[input_columns].values
target_data = data[target_columns].values

# 构建序列
X, Y = [], []
for i in range(len(data) - sequence_length):
    X.append(input_data[i: i + sequence_length])
    Y.append(target_data[i + sequence_length])
X = np.stack(X)
Y = np.stack(Y)
# 对称拼接
X = np.concatenate([X, -X], axis=2)

# 划分训练/测试/验证集
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=123, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.25, random_state=123, shuffle=False
)

num_steps, num_dims = X_train.shape[1], X_train.shape[2]
num_targets = len(target_columns)
print(f'Using {num_dims} dims, {num_steps} steps, {num_targets} targets')

# 数据标准化
scaler_X = preprocessing.StandardScaler().fit(X_train.reshape(-1, num_dims))
scaler_y = preprocessing.StandardScaler().fit(y_train)
X_train = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1, num_steps, num_dims)
X_val   = scaler_X.transform(X_val.reshape(-1, num_dims)).reshape(-1, num_steps, num_dims)
X_test  = scaler_X.transform(X_test.reshape(-1, num_dims)).reshape(-1, num_steps, num_dims)
y_train_scaled = scaler_y.transform(y_train)
y_val_scaled   = scaler_y.transform(y_val)

print('Input mean:', scaler_X.mean_)
print('Input std:', scaler_X.scale_)
print('Output mean:', scaler_y.mean_)
print('Output std:', scaler_y.scale_)

# -------------------------------------------------
# 数据集与加载器
# -------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train_scaled), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TimeSeriesDataset(X_val,   y_val_scaled),   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TimeSeriesDataset(X_test),                batch_size=batch_size, shuffle=False)

# -------------------------------------------------
# 从头实现传统 Transformer
# -------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = attn @ V  # (B, heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(context)
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (i / d_model)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (i / d_model)))
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe.unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.mha(x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class TransformerModel(nn.Module):
    def __init__(self, seq_len, num_dims, num_targets,
                 d_model=64, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_dims, d_model)
        self.pos_enc = PositionalEncoding(seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, num_targets)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]
        return self.head(x)

# -------------------------------------------------
# 实例化模型、损失和优化器
# -------------------------------------------------
model = TransformerModel(
    seq_len=num_steps,
    num_dims=num_dims,
    num_targets=num_targets,
    d_model=64,
    num_heads=1,
    ff_dim=64,
    num_layers=1,
    dropout=0.1
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------------------------------------
# 训练循环 (含 Early Stopping 和 Val Loss 打印)
# -------------------------------------------------
best_val_loss = float('inf')
patience = 5
wait = 0
train_losses, val_losses = [], []
start_time = time.time()
for epoch in range(1, epochs+1):
    # 训练
    model.train()
    train_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * Xb.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f}s, Epochs: {epoch}")

# 加载最优模型
model.load_state_dict(torch.load(model_path))

# -------------------------------------------------
# 测试评估
# -------------------------------------------------
model.eval()
all_preds = []
with torch.no_grad():
    for Xb in test_loader:
        Xb = Xb.to(device)
        preds = model(Xb).cpu().numpy()
        all_preds.append(preds)
y_pred_scaled = np.vstack(all_preds)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test, y_pred_real, multioutput='raw_values')
mape = mean_absolute_percentage_error(y_test, y_pred_real, multioutput='raw_values')
r2 = r2_score(y_test, y_pred_real, multioutput='raw_values')

print('Test MSE:', mse)
print('Test MAPE:', mape)
print('Test R2:', r2)

# 保存测试结果
with open(test_result_path, 'w') as f:
    f.write(f"Test MSE: {mse}\nMSE Avg: {mse.mean()}\n")
    f.write(f"Test MAPE: {mape}\nMAPE Avg: {mape.mean()}\n")
    f.write(f"Test R2: {r2}\nR2 Avg: {r2.mean()}\n")
    f.write(f"Training time: {end_time - start_time:.2f}s\n")
    f.write(f"Epochs: {epoch}\n")
# -------------------------------------------------
# 可视化损失和预测
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
    plt.savefig(os.path.join(visualization_path, f'Transformer_custom_comparison_{col}.png'))
