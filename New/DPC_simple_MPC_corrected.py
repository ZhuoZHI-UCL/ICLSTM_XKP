import time
import random
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchinfo import summary

# ============================================
# 设置随机种子以确保结果可复现
# ============================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# 特征定义：16 状态 + 6 控制 = 22 维输入
# ============================================
all_features = [
    'Bd_T_HP_supply', 'Bd_T_HP_return',
    'Z01_T','Z02_T','Z03_T','Z04_T','Z05_T','Z06_T','Z07_T','Z08_T',
    'Bd_FracCh_Bat','Fa_ECh_Bat','Fa_EDCh_Bat','Fa_Pw_Prod','PV_Gen_corrected','Fa_E_All',
    'P1_T_Thermostat_sp_out','P2_T_Thermostat_sp_out',
    'P3_T_Thermostat_sp_out','P4_T_Thermostat_sp_out',
    'Bd_Pw_Bat_sp_out','Bd_T_HP_sp_out'
]
states_features   = all_features[:16]   # fx 模型预测的 16 维状态
controls_features = all_features[16:]   # fu 模型预测的 6 维控制

num_states   = len(states_features)    # =16
num_controls = len(controls_features)  # =6
input_size   = num_states + num_controls  # =22

# ============================================
# 读入 CSV 并归一化
# ============================================
df = pd.read_csv(r'C:\Users\uceekx0\Desktop\ICLSTM_XKP\simulation_output.csv')
df = df[all_features]

states_data   = df[states_features].values.astype(np.float32)
controls_data = df[controls_features].values.astype(np.float32)

states_mean   = states_data.mean(axis=0)
states_std    = states_data.std(axis=0) + 1e-8
controls_mean = controls_data.mean(axis=0)
controls_std  = controls_data.std(axis=0) + 1e-8

# normalize
norm_states   = (states_data   - states_mean)   / states_std
norm_controls = (controls_data - controls_mean) / controls_std

# concatenate to shape (T, 22)
data_all = np.concatenate([norm_states, norm_controls], axis=1)

# ============================================
# 超参数
# ============================================
window_size         = 10
prediction_horizon  = 5
batch_size          = 256
num_epochs          = 200
learning_rate       = 0.005
interval_test       = 5

hidden_size_fx      = 32
hidden_size_fu      = 32

w_obj  = 1.0    # 目标惩罚权重
w_cons = 1e2    # 违反约束惩罚权重
w_id   = 1.0    # 识别误差权重
w_DPC  = 5.0    # DPC 总损失权重

# ============================================
# 约束阈值（真实值域）与归一化
# ============================================
# Z01_T - Z08_T 对应 states_features[2:10]
Z_T_min = 19.0; Z_T_max = 24.0
Z_min_norm = torch.tensor(
    [(Z_T_min - states_mean[i]) / states_std[i] for i in range(2,10)],
    dtype=torch.float32, device=device)
Z_max_norm = torch.tensor(
    [(Z_T_max - states_mean[i]) / states_std[i] for i in range(2,10)],
    dtype=torch.float32, device=device)
# 控制前4个 P1-P4
P_min = 15.0; P_max = 30.0
P_min_norm = torch.tensor(
    [(P_min - controls_mean[i]) / controls_std[i] for i in range(4)],
    dtype=torch.float32, device=device)
P_max_norm = torch.tensor(
    [(P_max - controls_mean[i]) / controls_std[i] for i in range(4)],
    dtype=torch.float32, device=device)

# ============================================
# 数据集定义
# ============================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.n_samples = len(data) - window_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        y = self.data[idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

full_ds = TimeSeriesDataset(data_all, window_size)
n_train = int(len(full_ds) * 0.7)
train_ds = Subset(full_ds, range(n_train))
test_ds  = Subset(full_ds, range(n_train, len(full_ds)))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ============================================
# 手写 LSTMCell 和模型
# ============================================
class StandardLSTMCell(nn.Module):
    def __init__(self, in_sz, hid_sz):
        super().__init__()
        # weights for i,f,o,g gates
        self.Wi = nn.Parameter(torch.randn(hid_sz, in_sz)*0.01)
        self.Ui = nn.Parameter(torch.randn(hid_sz, hid_sz)*0.01)
        self.bi = nn.Parameter(torch.zeros(hid_sz))
        self.Wf = nn.Parameter(torch.randn(hid_sz, in_sz)*0.01)
        self.Uf = nn.Parameter(torch.randn(hid_sz, hid_sz)*0.01)
        self.bf = nn.Parameter(torch.zeros(hid_sz))
        self.Wo = nn.Parameter(torch.randn(hid_sz, in_sz)*0.01)
        self.Uo = nn.Parameter(torch.randn(hid_sz, hid_sz)*0.01)
        self.bo = nn.Parameter(torch.zeros(hid_sz))
        self.Wc = nn.Parameter(torch.randn(hid_sz, in_sz)*0.01)
        self.Uc = nn.Parameter(torch.randn(hid_sz, hid_sz)*0.01)
        self.bc = nn.Parameter(torch.zeros(hid_sz))

    def forward(self, x, states):
        h_prev, c_prev = states
        xi = F.linear(x, self.Wi)
        ui = F.linear(h_prev, self.Ui, self.bi)
        xf = F.linear(x, self.Wf)
        uf = F.linear(h_prev, self.Uf, self.bf)
        xo = F.linear(x, self.Wo)
        uo = F.linear(h_prev, self.Uo, self.bo)
        xc = F.linear(x, self.Wc)
        uc = F.linear(h_prev, self.Uc, self.bc)
        i = torch.sigmoid(xi + ui)
        f = torch.sigmoid(xf + uf)
        o = torch.sigmoid(xo + uo)
        g = torch.tanh(xc + uc)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class TwoLayerLSTMModel(nn.Module):
    def __init__(self, in_sz, hid_sz, out_sz):
        super().__init__()
        self.lstm1 = StandardLSTMCell(in_sz, hid_sz)
        self.fc_b  = nn.Linear(hid_sz, in_sz)
        self.lstm2 = StandardLSTMCell(in_sz, hid_sz)
        self.fc_out = nn.Linear(hid_sz, out_sz)

    def forward(self, x):
        b, seq_len, _ = x.size()
        h1 = torch.zeros(b, self.lstm1.Ui.size(0), device=device)
        c1 = torch.zeros_like(h1)
        outs1 = []
        for t in range(seq_len):
            h1, c1 = self.lstm1(x[:,t,:], (h1,c1))
            outs1.append(h1.unsqueeze(1))
        seq1 = torch.cat(outs1, dim=1)         # [b, seq, hid]
        res  = F.relu(self.fc_b(seq1))         # [b, seq, in_sz]
        h2 = torch.zeros(b, self.lstm2.Ui.size(0), device=device)
        c2 = torch.zeros_like(h2)
        outs2 = []
        for t in range(seq_len):
            h2, c2 = self.lstm2(res[:,t,:], (h2,c2))
            outs2.append(h2.unsqueeze(1))
        seq2 = torch.cat(outs2, dim=1)         # [b, seq, hid]
        last = seq2[:,-1,:]                    # [b, hid]
        return self.fc_out(last)               # [b, out_sz]

class MLPModel(nn.Module):
    def __init__(self, in_sz, hid_sz, out_sz):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, hid_sz)
        self.fc2 = nn.Linear(hid_sz, out_sz)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# instantiate models
model_fx = TwoLayerLSTMModel(in_sz=input_size, hid_sz=hidden_size_fx, out_sz=num_states).to(device)
model_fu = MLPModel(in_sz=num_states, hid_sz=hidden_size_fu, out_sz=num_controls).to(device)

# optimizer & loss
optimizer = optim.Adam(list(model_fx.parameters()) + list(model_fu.parameters()), lr=learning_rate)
mse_loss  = nn.MSELoss()

def range_penalty(pred, lower, upper):
    return (torch.relu(lower - pred)**2 + torch.relu(pred - upper)**2)

# ============================================
# 评估函数（滚动 DPC 计算）
# ============================================
def evaluate_model(fx, fu, loader):
    fx.eval(); fu.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # supervise: first 16 dims
            target = y_batch[:, :num_states]
            fx_pred = fx(x_batch)
            loss_id = mse_loss(fx_pred, target)
            # DPC loop
            loss_DPC = 0.0
            window = x_batch.clone()
            for _ in range(prediction_horizon):
                latest = window[:, -1, :]
                st = latest[:, :num_states]
                ctrl_pred = fu(st)
                new_step = torch.cat([st, ctrl_pred], dim=1).unsqueeze(1)
                window = torch.cat([window[:,1:,:], new_step], dim=1)
                fx_new = fx(window)
                # constraint losses
                c1 = range_penalty(fx_new[:,2:10], Z_min_norm, Z_max_norm).mean()
                c2 = range_penalty(ctrl_pred[:,:4], P_min_norm, P_max_norm).mean()
                obj = (fx_new[:,13]**2).sum()
                loss_DPC += w_cons*(c1+c2) + w_obj*obj
            loss_DPC = loss_DPC / prediction_horizon
            loss = w_id*loss_id + w_DPC*loss_DPC
            total += loss.item(); count += 1
    fx.train(); fu.train()
    return total / count

# ============================================
# 训练循环
# ============================================
train_losses, test_losses, test_epochs = [], [], []
start_time = time.time()

for epoch in range(1, num_epochs+1):
    epoch_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        target = y_batch[:, :num_states]
        fx_pred = model_fx(x_batch)
        loss_id = mse_loss(fx_pred, target)
        loss_DPC = 0.0
        window = x_batch.clone()
        for _ in range(prediction_horizon):
            latest = window[:, -1, :]
            st = latest[:, :num_states]
            ctrl_pred = model_fu(st)
            new_step = torch.cat([st, ctrl_pred], dim=1).unsqueeze(1)
            window = torch.cat([window[:,1:,:], new_step], dim=1)
            fx_new = model_fx(window)
            c1 = range_penalty(fx_new[:,2:10], Z_min_norm, Z_max_norm).mean()
            c2 = range_penalty(ctrl_pred[:,:4], P_min_norm, P_max_norm).mean()
            obj = (fx_new[:,13]**2).sum()
            loss_DPC += w_cons*(c1+c2) + w_obj*obj
        loss_DPC = loss_DPC / prediction_horizon
        loss = w_id*loss_id + w_DPC*loss_DPC
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_train = epoch_loss / len(train_loader)
    train_losses.append(avg_train)
    print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train:.6f}")

    if epoch % interval_test == 0:
        t_loss = evaluate_model(model_fx, model_fu, test_loader)
        test_losses.append(t_loss)
        test_epochs.append(epoch)
        print(f"    >>> Test Loss at epoch {epoch}: {t_loss:.6f}")

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f}s, Avg per epoch: {total_time/num_epochs:.2f}s")

# model summaries
summary(model_fx, input_size=(batch_size, window_size, input_size))
summary(model_fu, input_size=(batch_size, num_states))

# ============================================
# 保存模型
# ============================================
base_dir = r"C:\Users\uceekx0\Desktop\ICLSTM_XKP\DPC\DPC_lstm_visualization"
os.makedirs(base_dir, exist_ok=True)
fx_path = os.path.join(base_dir, "DPC_lstm_fx_model.h5")
fu_path = os.path.join(base_dir, "DPC_lstm_fu_model.h5")
torch.save(model_fx.state_dict(), fx_path)
torch.save(model_fu.state_dict(), fu_path)
print(f"Saved fx to {fx_path}")
print(f"Saved fu to {fu_path}")

# ============================================
# 绘制损失曲线
# ============================================
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss", marker='o')
plt.plot(test_epochs, test_losses,     label="Test Loss",  marker='s', linestyle='--')
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training & Test Loss")
plt.grid(True); plt.legend()
plt.savefig("loss_DPC.png")
plt.show()
