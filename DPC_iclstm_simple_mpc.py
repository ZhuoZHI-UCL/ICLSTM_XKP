import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchinfo import summary

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

# 特征名称
all_features = [
    'Bd_T_HP_supply', 'Bd_T_HP_return',
    'Z01_T','Z02_T','Z03_T','Z04_T','Z05_T','Z06_T','Z07_T','Z08_T',
    'Bd_FracCh_Bat','Fa_Pw_Prod','PV_Gen_corrected','Fa_E_All',
    'P1_T_Thermostat_sp_out','P2_T_Thermostat_sp_out',
    'P3_T_Thermostat_sp_out','P4_T_Thermostat_sp_out',
    'Bd_Pw_Bat_sp_out','Bd_T_HP_sp_out'
]
states_features = all_features[:14]
controls_features = all_features[14:]

# ============================================
# 读取与归一化数据
# ============================================
data_df = pd.read_csv(r'simulation_output.csv')
data_df = data_df[all_features]

states_data = data_df[states_features].values.astype(np.float32)
controls_data = data_df[controls_features].values.astype(np.float32)

states_mean = states_data.mean(axis=0)
states_std = states_data.std(axis=0) + 1e-8
controls_mean = controls_data.mean(axis=0)
controls_std = controls_data.std(axis=0) + 1e-8

# 分别归一化 states 与 controls 数据，再拼接为 20 维数据（前 14 列 states, 后 6 列 controls）
normalized_states = (states_data - states_mean) / states_std
normalized_controls = (controls_data - controls_mean) / controls_std
normalized_data = np.concatenate([normalized_states, normalized_controls], axis=1)
normalized_data_ext = np.concatenate([normalized_data, -normalized_data], axis=1)

# ============================================
# 超参数设置
# ============================================
window_size = 10             # 初始时间序列长度
prediction_horizon = 2      # 滚动预测步数
batch_size = 256             # mini-batch 大小
num_epochs = 200             # 总训练轮数
learning_rate = 0.005        # 学习率
interval_test = 5            # 测试间隔

# 模型参数
hidden_size_ic = 32          # ICLSTM 隐藏层维度
hidden_size_fu = 32          # MLP 隐藏层维度

# 损失权重
w_obj = 1.0  # fx 中 Fa_E_All 项的目标惩罚权重
w_cons = 1e3  # 违反约束的惩罚权重（fx 和 fu 约束均合并）
w_id   = 1.0  # fx 识别误差损失权重（监督部分）
w_DPC  = 5.0  # 平衡 fx 识别与 fu（policy）性能的权重（DPC 损失）

# 约束阈值（真实数值域）
Z_T_min = 19.0
Z_T_max = 24.0
P_T_Thermostat_sp_min = 15.0
P_T_Thermostat_sp_max = 30.0
# --------------------------------------------
# 根据原始约束阈值及归一化参数，计算归一化后的阈值
# --------------------------------------------
# f(x): 对应 'Z01_T'-'Z08_T' 在 all_features 中的索引为 2 ~ 9
Z_T_min_norm = torch.tensor([(Z_T_min - states_mean[i]) / states_std[i] for i in range(2, 10)], dtype=torch.float32)
Z_T_max_norm = torch.tensor([(Z_T_max - states_mean[i]) / states_std[i] for i in range(2, 10)], dtype=torch.float32)

# f(u): 对应 f(u) 前 4 个特征，在 all_features 中的索引为 14,15,16,17
P_T_Thermostat_sp_min_norm = torch.tensor([(P_T_Thermostat_sp_min - controls_mean[i]) / controls_std[i] for i in range(0, 4)], dtype=torch.float32)
P_T_Thermostat_sp_max_norm = torch.tensor([(P_T_Thermostat_sp_max - controls_mean[i]) / controls_std[i] for i in range(0, 4)], dtype=torch.float32)


# ============================================
# 时间序列数据集
# ============================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        """
        data: np.array, shape=(T, num_features) —— 此处 num_features=20
        window_size: 初始输入序列长度
        目标为紧跟 window_size 之后的下一个 timestep（单步预测）
        """
        self.data = data
        self.window_size = window_size
        self.num_samples = len(data) - window_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq = self.data[idx: idx + self.window_size]
        target_seq = self.data[idx + self.window_size]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)

# 构造数据集、训练集和测试集
full_dataset = TimeSeriesDataset(normalized_data_ext, window_size)
num_train = int(len(full_dataset) * 0.7)
train_dataset = Subset(full_dataset, range(num_train))
test_dataset = Subset(full_dataset, range(num_train, len(full_dataset)))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================
# 定义 ICLSTMCell 和 ICLSTMModel
# ============================================
class ICLSTMCellPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 注意: Wi 维度为 (hidden_size, input_size) 以匹配 F.linear 输入格式
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
    def forward(self, x, states):
        h_tm1, c_tm1 = states
        xi = F.linear(x, self.Wi)
        ui = F.linear(h_tm1, self.Ui)
        i = F.relu(self.DWi * (xi + ui) + self.bi)
        f = F.relu(self.DWf * (xi + ui) + self.bf)
        o = F.relu(self.DWo * (xi + ui) + self.bo)
        c_bar = F.relu(self.DWc * (xi + ui) + self.bc)
        c = f * c_tm1 + i * c_bar
        h = o * F.relu(c)
        return h, c
    def clamp_nonneg(self):
        for p in self.parameters(): p.data.clamp_(min=0.0)

class ICLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cell1 = ICLSTMCellPyTorch(input_size, hidden_size)
        self.cell2 = ICLSTMCellPyTorch(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.fc_out = nn.Linear(input_size, output_size)
    def forward(self, x):
        b, seq_len, _ = x.size()
        h1 = x.new_zeros(b, self.cell1.Ui.size(0))
        c1 = x.new_zeros(b, self.cell1.Ui.size(0))
        outs1 = []
        for t in range(seq_len):
            h1, c1 = self.cell1(x[:,t,:], (h1,c1))
            outs1.append(h1.unsqueeze(1))
        seq1 = torch.cat(outs1, dim=1)
        res1 = F.relu(self.fc1(seq1)) + x
        h2 = x.new_zeros(b, self.cell2.Ui.size(0))
        c2 = x.new_zeros(b, self.cell2.Ui.size(0))
        outs2 = []
        for t in range(seq_len):
            h2, c2 = self.cell2(res1[:,t,:], (h2,c2))
            outs2.append(h2.unsqueeze(1))
        seq2 = torch.cat(outs2, dim=1)
        res2 = F.relu(self.fc2(seq2)) + x
        out_seq = self.fc_out(res2)
        return out_seq[:, -1, :]
    def clamp_weights(self):
        self.cell1.clamp_nonneg()
        self.cell2.clamp_nonneg()
        self.fc1.weight.data.clamp_(min=0.0)
        self.fc2.weight.data.clamp_(min=0.0)
        self.fc_out.weight.data.clamp_(min=0.0)

# MLP policy 模型
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ============================================
# 初始化模型、优化器、损失
# ============================================
model_fx = ICLSTMModel(input_size=40, hidden_size=hidden_size_ic, output_size=14)
model_fu = MLPModel(input_size=14, hidden_size=hidden_size_fu, output_size=6)

optimizer = optim.Adam(list(model_fx.parameters()) + list(model_fu.parameters()), lr=learning_rate)

# ============================================
# 定义损失函数与约束损失函数
# ============================================
def range_penalty(pred, lower, upper):
    """
    pred: 反归一化后的预测值, 形状为 [batch, num_features]
    lower, upper: 原始阈值（标量或可广播形状）
    返回每个样本的损失（未做 mean 聚合，后续再取 mean）
    """
    loss_lower = torch.relu(lower - pred)
    loss_upper = torch.relu(pred - upper)
    return (loss_lower ** 2) + (loss_upper ** 2)

mse_loss = nn.MSELoss()

# ============================================
# 评估函数（滚动预测）
# ============================================
def evaluate_model(model_fx, model_fu, dataloader):
    model_fx.eval()
    model_fu.eval()
    total_loss = 0.0
    cnt = 0

    # 反归一化 states 监督部分：fx 模型输出 14 维
    states_mean_tensor_full = torch.tensor(states_mean, dtype=torch.float32)
    states_std_tensor_full  = torch.tensor(states_std, dtype=torch.float32)
    # 反归一化 states 约束部分：对应 states_features 的第 3 到第 10 列
    states_mean_tensor_cons = torch.tensor(states_mean[2:10], dtype=torch.float32)
    states_std_tensor_cons  = torch.tensor(states_std[2:10], dtype=torch.float32)
    # 反归一化 controls 约束部分：fu 模型输出 6 维，取前 4 维
    controls_mean_tensor_part = torch.tensor(controls_mean[0:4], dtype=torch.float32)
    controls_std_tensor_part  = torch.tensor(controls_std[0:4], dtype=torch.float32)

    with torch.no_grad():
        for batch_idx, (init_seq, target_seq) in enumerate(dataloader):
            # target_seq 形状为 [batch, 20]，监督目标取前 14 维
            target = target_seq[:, :14]
            # 通过初始窗口获得 fx 预测, 形状 [batch, 14]
            fx_pred = model_fx(init_seq)
            # 反归一化
            fx_pred_unnorm = fx_pred * states_std_tensor_full.to(fx_pred.device) + states_mean_tensor_full.to(fx_pred.device)
            target_unnorm = target * states_std_tensor_full.to(target.device) + states_mean_tensor_full.to(target.device)
            loss_id = mse_loss(fx_pred, target)

            loss_DPC = 0.0
            current_window = init_seq.clone()  # [batch, window_size, 20]

            for k in range(prediction_horizon):
                # start_time = time.time()
                latest_timestep = current_window[:, -1, :]      # [batch, 20]
                latest_state = latest_timestep[:, :14]           # 取前 14 维
                fu_pred_new = model_fu(latest_state)             # [batch, 6]
                new_timestep = torch.cat([latest_state, fu_pred_new], dim=1)  # [batch, 20]
                new_timestep_ext = torch.cat([new_timestep, -new_timestep], dim=1).unsqueeze(1)
                new_window = torch.cat([current_window[:, :-1, :], new_timestep_ext], dim=1)
                new_fx_pred = model_fx(new_window)           # [batch, 14]
                # 反归一化
                new_fx_pred_unnorm = new_fx_pred * states_std_tensor_full.to(new_fx_pred.device) + states_mean_tensor_full.to(new_fx_pred.device)
                loss_range_fx = range_penalty(new_fx_pred[:, 2:10], Z_T_min_norm.to(new_fx_pred.device), Z_T_max_norm.to(new_fx_pred.device)).mean()
                # 反归一化
                fu_pred_new_unnorm = fu_pred_new[:, :4] * controls_std_tensor_part.to(fu_pred_new.device) + controls_mean_tensor_part.to(fu_pred_new.device)
                loss_range_fu = range_penalty(fu_pred_new[:, :4], P_T_Thermostat_sp_min_norm.to(fu_pred_new.device), P_T_Thermostat_sp_max_norm.to(fu_pred_new.device)).mean()
                loss_cons = loss_range_fx + loss_range_fu
                loss_obj = (new_fx_pred[:, 13] ** 2).sum()
                loss_DPC += w_cons * loss_cons + w_obj * loss_obj
                # end_time = time.time()
                # print(f"Time taken for step {k+1}: {end_time - start_time:.4f} seconds")
            loss_DPC = loss_DPC / prediction_horizon
            loss_val = w_id * loss_id + w_DPC * loss_DPC
            total_loss += loss_val.item()
            cnt += 1
    model_fx.train()
    model_fu.train()
    return total_loss / cnt

# ============================================
# 训练过程：每 interval_test 个 epoch 在测试集上评估一次
# ============================================
training_losses = []
test_losses = []
test_epochs = []

# 全局定义训练时用的反归一化张量（用于监督损失）：
states_mean_tensor_full = torch.tensor(states_mean, dtype=torch.float32)
states_std_tensor_full  = torch.tensor(states_std, dtype=torch.float32)
controls_mean_tensor_part = torch.tensor(controls_mean[0:4], dtype=torch.float32)
controls_std_tensor_part  = torch.tensor(controls_std[0:4], dtype=torch.float32)

model_fx.train()
model_fu.train()

# 记录训练开始时间
start_time = time.time()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (init_seq, target_seq) in enumerate(train_loader):
        optimizer.zero_grad()
        # target_seq 形状为 [batch, 20]，监督目标取前 14 维
        target = target_seq[:, :14]
        fx_pred = model_fx(init_seq)  # [batch, 14]
        # 反归一化
        fx_pred_unnorm = fx_pred * states_std_tensor_full.to(fx_pred.device) + states_mean_tensor_full.to(fx_pred.device)
        target_unnorm = target * states_std_tensor_full.to(target.device) + states_mean_tensor_full.to(target.device)
        loss_id = mse_loss(fx_pred, target)

        loss_DPC = 0.0
        current_window = init_seq.clone()  # [batch, window_size, 20]

        for k in range(prediction_horizon):
            latest_timestep = current_window[:, -1, :]    # [batch, 40]
            latest_state = latest_timestep[:, :14]      # [batch, 14]
            fu_pred_new = model_fu(latest_state)            # [batch, 6]
            new_timestep = torch.cat([latest_state, fu_pred_new], dim=1)  # [batch, 20]
            new_timestep_ext = torch.cat([new_timestep, -new_timestep], dim=1).unsqueeze(1)
            new_window = torch.cat([current_window[:, :-1, :], new_timestep_ext], dim=1)
            new_fx_pred = model_fx(new_window)  # [batch, 14]
            # 反归一化
            new_fx_pred_unnorm = new_fx_pred * states_std_tensor_full.to(new_fx_pred.device) + states_mean_tensor_full.to(new_fx_pred.device)
            fu_pred_new_unnorm = fu_pred_new[:, :4] * controls_std_tensor_part.to(fu_pred_new.device) + controls_mean_tensor_part.to(fu_pred_new.device)

            loss_range_fx = range_penalty(new_fx_pred[:, 2:10], Z_T_min_norm.to(new_fx_pred.device),
                                          Z_T_max_norm.to(new_fx_pred.device)).mean()

            loss_range_fu = range_penalty(fu_pred_new[:, :4], P_T_Thermostat_sp_min_norm.to(fu_pred_new.device),
                                          P_T_Thermostat_sp_max_norm.to(fu_pred_new.device)).mean()

            loss_cons = loss_range_fx + loss_range_fu

            loss_obj = (new_fx_pred[:, 13] ** 2).mean()

            # loss_DPC += w_cons * loss_cons + w_obj * loss_obj
            loss_DPC += loss_cons + loss_obj
        loss_DPC = loss_DPC / prediction_horizon
        loss_val = w_id * loss_id + w_DPC * loss_DPC
        start_time = time.time()
        loss_val.backward()
        end_time = time.time()
        print(f"Time taken for backward: {end_time - start_time:.4f} seconds")
        optimizer.step()
        model_fx.clamp_weights()  # 新增约束
        epoch_loss += loss_val.item()

    avg_train_loss = epoch_loss / len(train_loader)
    training_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}")

    if (epoch+1) % interval_test == 0:
        test_loss = evaluate_model(model_fx, model_fu, test_loader)
        test_losses.append(test_loss)
        test_epochs.append(epoch+1)
        print(f"    >>> Test Loss at epoch {epoch+1}: {test_loss:.6f}")
# 记录训练结束时间，并计算训练总耗时
end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")
# summary(model_fx, input_size=(batch_size, 10, 20))
# summary(model_fu, input_size=(batch_size, 1, 14))

# ============================================
# 绘制损失曲线
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), training_losses, label="Train Loss", marker="o")
plt.plot(test_epochs, test_losses, label="Test Loss", marker="s", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.savefig("loss_DPC.png")
plt.show()

