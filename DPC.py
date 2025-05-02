import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
# ============================================
# 设置随机种子以确保结果可复现
# ============================================
seed = 42  # 你可以根据需要设定任意固定的整数值
random.seed(seed)            # 设置 Python 内置 random 的种子
np.random.seed(seed)         # 设置 numpy 的随机种子
torch.manual_seed(seed)      # 设置 PyTorch CPU 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU，设置所有GPU的随机种子
# 为了确保 CuDNN 使用确定性的算法（这样会略微影响性能，但保证结果一致）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================
# 超参数设置（包括约束相关参数）
# ============================================
window_size = 10         # 初始时间序列长度（step）
roll_steps = 2          # 滚动预测的步数（N）
batch_size = 256          # mini-batch 大小
num_epochs = 200          # 总训练轮数
learning_rate = 0.001    # 学习率
interval_test = 5         # 每多少个 epoch 测试一次
# 模型参数
hidden_size_fx = 32      # LSTM 的隐藏层维度
num_layers_fx = 1        # LSTM 层数
hidden_size_fu = 32     # MLP 隐藏层维度

# -----------------------------
# 约束损失的权重参数（方便调整）
# -----------------------------
lambda_fa = 0.1          # f(x) 中 Fa_E_All 越小越好的惩罚项权重
lambda_range_fx = 0.1    # f(x) 中 'Z01_T'-'Z08_T' 超出区间（原始值 19-24）的惩罚项权重
lambda_range_fu = 0.1    # f(u) 中前4个特征超出区间（原始值 15-30）的惩罚项权重

# -----------------------------
# 原始约束阈值（真实数值域）
# -----------------------------
fx_range_lower = 19.0    # f(x)中 'Z01_T'-'Z08_T' 的下界
fx_range_upper = 24.0    # f(x)中 'Z01_T'-'Z08_T' 的上界
fu_range_lower = 15.0    # f(u)中前4个特征的下界
fu_range_upper = 30.0    # f(u)中前4个特征的上界

# ============================================
# 定义各特征名称
# ============================================
all_features = ['Bd_T_HP_supply', 'Bd_T_HP_return', 
                'Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 'Z06_T', 'Z07_T', 'Z08_T', 
                'Bd_FracCh_Bat', 'Fa_Pw_Prod', 'PV_Gen_corrected', 'Fa_E_All', 
                'P1_T_Thermostat_sp_out', 'P2_T_Thermostat_sp_out', 
                'P3_T_Thermostat_sp_out', 'P4_T_Thermostat_sp_out', 
                'Bd_Pw_Bat_sp_out', 'Bd_T_HP_sp_out']

# f(x) 预测的 14 个特征（顺序与原始数据中保持一致）
fx_features = ['Bd_T_HP_supply', 'Bd_T_HP_return', 
               'Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 'Z06_T', 'Z07_T', 'Z08_T', 
               'Bd_FracCh_Bat', 'Fa_Pw_Prod', 'PV_Gen_corrected', 'Fa_E_All']

# f(u) 预测的 6 个特征
fu_features = ['P1_T_Thermostat_sp_out', 'P2_T_Thermostat_sp_out', 
               'P3_T_Thermostat_sp_out', 'P4_T_Thermostat_sp_out', 
               'Bd_Pw_Bat_sp_out', 'Bd_T_HP_sp_out']

# ============================================
# 读取 CSV 数据并归一化处理
# ============================================
data_df = pd.read_csv('./simulation_output.csv')
data_df = data_df[all_features]  # 保持特征顺序一致
data_values = data_df.values.astype(np.float32)

# 均值-标准差归一化（记录均值与标准差，便于将约束阈值转换到归一化空间）
data_mean = data_values.mean(axis=0)
data_std = data_values.std(axis=0) + 1e-8  # 防止除0
data_norm = (data_values - data_mean) / data_std

# --------------------------------------------
# 根据原始约束阈值及归一化参数，计算归一化后的阈值
# --------------------------------------------
# f(x): 对应 'Z01_T'-'Z08_T' 在 all_features 中的索引为 2 ~ 9
fx_norm_lower = torch.tensor([(fx_range_lower - data_mean[i]) / data_std[i] for i in range(2, 10)], dtype=torch.float32)
fx_norm_upper = torch.tensor([(fx_range_upper - data_mean[i]) / data_std[i] for i in range(2, 10)], dtype=torch.float32)

# f(u): 对应 f(u) 前 4 个特征，在 all_features 中的索引为 14,15,16,17
fu_norm_lower = torch.tensor([(fu_range_lower - data_mean[i]) / data_std[i] for i in [14,15,16,17]], dtype=torch.float32)
fu_norm_upper = torch.tensor([(fu_range_upper - data_mean[i]) / data_std[i] for i in [14,15,16,17]], dtype=torch.float32)

# ============================================
# 定义时间序列数据集
# ============================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, roll_steps):
        """
        data: np.array, shape=(T, num_features)
        window_size: 初始输入序列长度
        roll_steps: 滚动预测的步数
        """
        self.data = data
        self.window_size = window_size
        self.roll_steps = roll_steps
        self.seq_length = window_size + roll_steps
        self.num_samples = len(data) - self.seq_length + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_length]
        input_seq = seq[:self.window_size]
        target_seq = seq[self.window_size:]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)

# 构造整个数据集
full_dataset = TimeSeriesDataset(data_norm, window_size, roll_steps)
num_total = len(full_dataset)
num_train = int(num_total * 0.7)  # 70% 用于训练
num_test = num_total - num_train

# 划分训练集和测试集（按时间顺序划分，更适合时序数据）
train_dataset = Subset(full_dataset, range(num_train))
test_dataset  = Subset(full_dataset, range(num_train, num_total))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================
# 定义模型
# ============================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)         # out: [batch, seq_len, hidden_size]
        last_out = out[:, -1, :]      # 取最后一个时间步的输出
        output = self.fc(last_out)    # 映射到 output_size 维
        return output

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model_fx = LSTMModel(input_size=20, hidden_size=hidden_size_fx, num_layers=num_layers_fx, output_size=14)
model_fu = MLPModel(input_size=14, hidden_size=hidden_size_fu, output_size=6)

# ============================================
# 定义损失函数与约束损失函数
# ============================================
def range_penalty(pred, lower, upper):
    """
    pred: [batch, num_features]
    lower, upper: 可为标量或者形状为 [num_features] 的张量
    """
    loss_lower = torch.relu(lower - pred)
    loss_upper = torch.relu(pred - upper)
    return ((loss_lower ** 2) + (loss_upper ** 2)).mean()

mse_loss = nn.MSELoss()

# ============================================
# 定义优化器（同时更新两个模型的参数）
# ============================================
optimizer = optim.Adam(list(model_fx.parameters()) + list(model_fu.parameters()), lr=learning_rate)

# ============================================
# 定义测试/评估函数（滚动预测方式计算 loss）
# ============================================
def evaluate_model(model_fx, model_fu, dataloader):
    model_fx.eval()
    model_fu.eval()
    total_loss = 0.0
    cnt = 0
    with torch.no_grad():
        for batch_idx, (init_seq, target_seq) in enumerate(dataloader):
            current_window = init_seq.clone()  # [batch, window_size, 20]
            batch_loss = 0.0
            for t in range(roll_steps):
                fx_pred = model_fx(current_window)        # [batch, 14]
                fu_pred = model_fu(fx_pred)                 # [batch, 6]
                full_pred = torch.cat([fx_pred, fu_pred], dim=1)  # [batch, 20]
                target = target_seq[:, t, :]                # [batch, 20]
                loss_full = mse_loss(full_pred, target)
                loss_fx = mse_loss(fx_pred, target[:, :14])
                loss_fu = mse_loss(fu_pred, target[:, 14:])
                loss_fa = (fx_pred[:, 13] ** 2).mean()
                loss_range_fx = range_penalty(fx_pred[:, 2:10],
                                              fx_norm_lower.to(fx_pred.device),
                                              fx_norm_upper.to(fx_pred.device))
                loss_range_fu = range_penalty(fu_pred[:, :4],
                                              fu_norm_lower.to(fu_pred.device),
                                              fu_norm_upper.to(fu_pred.device))
                loss = loss_full + loss_fx + loss_fu \
                       + lambda_fa * loss_fa \
                       + lambda_range_fx * loss_range_fx \
                       + lambda_range_fu * loss_range_fu
                batch_loss += loss.item()
                # 滚动更新当前窗口
                current_window = torch.cat([current_window[:, 1:, :], full_pred.unsqueeze(1)], dim=1)
            total_loss += batch_loss
            cnt += 1
    model_fx.train()
    model_fu.train()
    return total_loss / cnt  # 平均每个 batch 的累计 loss

# ============================================
# 训练过程：每  interval_test 个 epoch 在测试集上评估一次
# ============================================
training_losses = []
test_losses = []
test_epochs = []  # 记录测试时对应的 epoch

model_fx.train()
model_fu.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (init_seq, target_seq) in enumerate(train_loader):
        optimizer.zero_grad()
        current_window = init_seq.clone()  # [batch, window_size, 20]
        loss_total = 0.0
        for t in range(roll_steps):
            fx_pred = model_fx(current_window)         # [batch, 14]
            fu_pred = model_fu(fx_pred)                  # [batch, 6]
            full_pred = torch.cat([fx_pred, fu_pred], dim=1)  # [batch, 20]
            target = target_seq[:, t, :]                 # [batch, 20]
            loss_full = mse_loss(full_pred, target)
            loss_fx = mse_loss(fx_pred, target[:, :14])
            # loss_fu = mse_loss(fu_pred, target[:, 14:])
            loss_fa = (fx_pred[:, 13] ** 2).mean()
            loss_range_fx = range_penalty(fx_pred[:, 2:10],
                                          fx_norm_lower.to(fx_pred.device),
                                          fx_norm_upper.to(fx_pred.device))
            loss_range_fu = range_penalty(fu_pred[:, :4],
                                          fu_norm_lower.to(fu_pred.device),
                                          fu_norm_upper.to(fu_pred.device))
            loss = loss_full + loss_fx +  \
                   + lambda_fa * loss_fa \
                   + lambda_range_fx * loss_range_fx \
                   + lambda_range_fu * loss_range_fu
            loss_total += loss
            current_window = torch.cat([current_window[:, 1:, :], full_pred.unsqueeze(1)], dim=1)
        
        loss_total.backward()
        optimizer.step()
        epoch_loss += loss_total.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    training_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}")
    
    # 每 interval_test 个 epoch 在测试集上评估一次
    if (epoch + 1) % interval_test == 0:
        test_loss = evaluate_model(model_fx, model_fu, test_loader)
        test_losses.append(test_loss)
        test_epochs.append(epoch + 1)
        print(f"    >>> Test Loss at epoch {epoch+1}: {test_loss:.6f}")

# ============================================
# 训练和测试损失可视化
# ============================================
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs+1), training_losses, label="Train Loss", marker="o")
plt.plot(test_epochs, test_losses, label="Test Loss", marker="s", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.savefig("loss_DPC.png")
plt.show()
