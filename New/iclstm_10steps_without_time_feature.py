import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Activation, Dropout, Add, LSTM, GRU, RNN, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras import backend as K
from keras.optimizers import Adam,SGD
import tensorflow as tf
from keras import Model, regularizers, activations, initializers
import pickle
from sklearn.preprocessing import StandardScaler
from ICLSTM import MyICLSTMCell
import time
import os
import random

# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf.random.set_seed(42)
# 读取数据
data = pd.read_csv("./New/simulation_output_whole_year.csv")
#保存的路径
visualization_path = 'visualization/iclstm_10steps_without_time_feature/'
test_result_path  = os.path.join(visualization_path, 'test_result.txt')
model_path        = os.path.join(visualization_path, 'iclstm_10steps_without_time_feature.h5')
scaler_stats_path = os.path.join(visualization_path, 'scaler_stats.npz')
# csv_paths = [
#     r"C:\Users\uceekx0\Desktop\ICLSTM_XKP\Generate_data_energym\simulation_output_Q1.csv",
#     r"C:\Users\uceekx0\Desktop\ICLSTM_XKP\Generate_data_energym\simulation_output_Q2.csv",
#     r"C:\Users\uceekx0\Desktop\ICLSTM_XKP\Generate_data_energym\simulation_output_Q3.csv",
#     r"C:\Users\uceekx0\Desktop\ICLSTM_XKP\Generate_data_energym\simulation_output_Q4.csv",
# ]

# 如果文件夹不存在，就创建
os.makedirs(visualization_path, exist_ok=True)
# 配置
sequence_length = 10
epochs = 200

input = ['Bd_T_HP_supply', 'Bd_T_HP_return', 'Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 'Z06_T', 'Z07_T', 'Z08_T', 'Bd_FracCh_Bat', 'Fa_ECh_Bat', 'Fa_EDCh_Bat', 'Fa_Pw_Prod', 'PV_Gen_corrected', 'Fa_E_All', 'P1_T_Thermostat_sp_out', 'P2_T_Thermostat_sp_out', 'P3_T_Thermostat_sp_out', 'P4_T_Thermostat_sp_out', 'Bd_Pw_Bat_sp_out', 'Bd_T_HP_sp_out']
predict_columns = ['Bd_T_HP_supply', 'Bd_T_HP_return', 'Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 'Z06_T', 'Z07_T', 'Z08_T', 'Bd_FracCh_Bat', 'Fa_ECh_Bat', 'Fa_EDCh_Bat', 'Fa_Pw_Prod', 'PV_Gen_corrected', 'Fa_E_All']

# 获取列索引
# 保持与 input 中的顺序一致
input_columns = [col for col in input if col in data.columns]
input_indices = [data.columns.get_loc(col) for col in input_columns]

# 同理，如果对 target_indices 也需要按 predict_columns 的顺序
target_columns = [col for col in predict_columns if col in data.columns]
target_indices = [data.columns.get_loc(col) for col in target_columns]
# # 去除目标变量构建输入特征索引
# input_columns = [col for col in data.columns if col  in input]
# input_indices = [data.columns.get_loc(col) for col in input_columns]
# target_indices = [data.columns.get_loc(col) for col in predict_columns]
# 转成 numpy
input_data = data[input_columns].values   # shape: (total_samples, num_input_feats)
target_data = data[predict_columns].values  # shape: (total_samples, 11)

# 构建数据集
X, Y = [], []
for i in range(len(data) - sequence_length):
    seq_input = input_data[i : i + sequence_length]  # (40, num_input_feats)
    seq_target = target_data[i + sequence_length]     # (11,)
    X.append(seq_input)
    Y.append(seq_target)

X = np.array(X)  # shape: (samples, 40, num_input_feats)
#注意X需要复制一份
X = np.concatenate((X,-X),axis=2)
Y = np.array(Y)
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=123,
                                                    shuffle=False)
num_dims = X_train.shape[2]
num_step = sequence_length
num_target = len(predict_columns)
print(f'we use {num_dims} features, {num_step} steps, {num_target} targets')
#归一化
scaler_X = preprocessing.StandardScaler().fit(X_train.reshape(-1, num_dims))
scaler_y = preprocessing.StandardScaler().fit(y_train.reshape(-1, num_target))
X_train = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
X_test = scaler_X.transform(X_test.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
y_train = scaler_y.transform(y_train.reshape(-1,num_target))
print("mean of input = ", scaler_X.mean_)
print("std of input = ", scaler_X.scale_)
print("mean of output = ", scaler_y.mean_)
print("std of output = ", scaler_y.scale_)
# =============== 保存 scaler 参数 ===============
np.savez(
    scaler_stats_path,
    input_mean  = scaler_X.mean_,
    input_std   = scaler_X.scale_,
    output_mean = scaler_y.mean_,
    output_std  = scaler_y.scale_,
)
print(f"Scaler stats saved to {scaler_stats_path}")
# ------------------------------------------

# # ---------------------------- 功能函数 ----------------------------
# def build_samples(df, in_cols, tgt_cols, seq_len):
#     """单个 DataFrame → 滑窗序列 (X, y)"""
#     X, y       = [], []
#     in_values  = df[in_cols ].values
#     tgt_values = df[tgt_cols].values
#     for i in range(len(df) - seq_len):
#         X.append(in_values [i : i + seq_len])
#         y.append(tgt_values[i + seq_len])
#     return np.asarray(X), np.asarray(y)
#
# # ------------------------- 1. 按文件构样本并切分 -------------------
# train_X_parts, train_y_parts = [], []
# test_X_parts , test_y_parts  = [], []
#
# for path in csv_paths:
#     print(f"Building samples from {path}")
#     df        = pd.read_csv(path)
#     # 以 Bd_T_HP_supply (或任意关键变量) 为基准：
#     first_valid = df.loc[df['Bd_T_HP_supply'] != 0].index[0]
#     df = df.loc[first_valid:].reset_index(drop=True)
#     # 若有多列需同时为 0，可换成 (df[zone_cols] == 0).all(axis=1)
#     Xi, Yi    = build_samples(df, input_columns, target_columns, sequence_length)
#     split_idx = int(len(Xi) * 0.7)           # 70 % 训练，30 % 测试（保持时序）
#
#     train_X_parts.append(Xi[:split_idx])
#     train_y_parts.append(Yi[:split_idx])
#     test_X_parts .append(Xi[split_idx:])
#     test_y_parts .append(Yi[split_idx:])
#     print(f"  -> total {len(Xi)} | train {split_idx} | test {len(Xi)-split_idx}")
#
# # ------------------------- 2. 合并四个季度 ------------------------
# X_train = np.concatenate(train_X_parts, axis=0)
# y_train = np.concatenate(train_y_parts, axis=0)
# X_test  = np.concatenate(test_X_parts , axis=0)
# y_test  = np.concatenate(test_y_parts , axis=0)
#
# print(f"Final train  sequences: {len(X_train)}")
# print(f"Final test   sequences: {len(X_test)}")
#
# # ------------------------- 3. 复制 −X 并拼接 -----------------------
# X_train = np.concatenate([X_train, -X_train], axis=2)
# X_test  = np.concatenate([X_test , -X_test ], axis=2)
#
# num_dims   = X_train.shape[2]
# num_step   = sequence_length
# num_target = y_train.shape[1]
# print(f'we use {num_dims} features, {num_step} steps, {num_target} targets')
#
# # ------------------------- 4. 标准化 (仅用训练集拟合) ---------------
# scaler_X = preprocessing.StandardScaler().fit(X_train.reshape(-1, num_dims))
# scaler_y = preprocessing.StandardScaler().fit(y_train.reshape(-1, num_target))
#
# X_train = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1, num_step, num_dims)
# X_test  = scaler_X.transform(X_test .reshape(-1, num_dims)).reshape(-1, num_step, num_dims)
# y_train = scaler_y.transform(y_train.reshape(-1, num_target))
#
# print("mean of input  = ", scaler_X.mean_)
# print("std  of input  = ", scaler_X.scale_)
# print("mean of output = ", scaler_y.mean_)
# print("std  of output = ", scaler_y.scale_)
#
# # ------------------------- 5. 保存 scaler 参数 ----------------------
# np.savez(
#     scaler_stats_path,
#     input_mean  = scaler_X.mean_,
#     input_std   = scaler_X.scale_,
#     output_mean = scaler_y.mean_,
#     output_std  = scaler_y.scale_,
# )
# print(f"Scaler stats saved to {scaler_stats_path}")


# 定义Convex网络
# ICLSTM
input = Input(shape=(X_train.shape[1],X_train.shape[2]))
x_skip = input
x = RNN(MyICLSTMCell(units=128),return_sequences=True)(input)
x = Dense(X_train.shape[2], activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = Add()([x, x_skip])


x = RNN(MyICLSTMCell(units=128),return_sequences=True)(x) #只输出最后一个预测的值
x = Dense(X_train.shape[2], activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = Add()([x, x_skip])
x = Dense(num_target, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = x[:,-1,:]
x = tf.reshape(x, (-1, num_target)) 
model = Model(input, x)

# 训练配置
# 1. 定义一个学习率衰减调度器
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
# lr_schedule = ExponentialDecay(
#     initial_learning_rate=1e-3,  # 初始学习率
#     decay_steps=10000,           # 每隔多少 step 衰减一次
#     decay_rate=0.1,             # 衰减系数(例如衰减到原来的96%)
#     staircase=True               # True表示阶梯衰减，False表示连续指数衰减
# )
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',       # 监控验证集损失
    patience=10,              # 如果10个epoch内val_loss没有改善，则停止训练
    restore_best_weights=True  # 恢复训练过程中表现最好的模型权重
)
optimizer = Adam(learning_rate=1e-4, clipnorm=None)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
start_time = time.time()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=256, validation_split=0.25, verbose=2, callbacks=[early_stopping])
end_time = time.time()
total_time = end_time - start_time
print(f"训练耗时: {total_time:.2f} 秒")
model.summary()
total_epochs = len(history.history['loss'])
model.save(model_path)

# 预测 & 反归一化
from sklearn.metrics import mean_squared_error, r2_score
y_pred_scaled = model.predict(X_test)  # (num_test_samples, 11)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
mse = mean_squared_error(y_test, y_pred_real, multioutput='raw_values')
mape = mean_absolute_percentage_error(y_test, y_pred_real, multioutput='raw_values')
r2 = r2_score(y_test, y_pred_real,multioutput='raw_values')

print("Test MSE:", mse)
print("Test MAPE:", mape)
print("Test R2:", r2)
#保存结果到test_result.txt 直接覆盖之前的结果
with open(test_result_path, 'w') as f:
    f.write('Test MSE: {}\n'.format(mse))
    f.write('MSE Average: {}\n'.format(np.mean(mse)))
    
    f.write('Test MAPE: {}\n'.format(mape))
    f.write('MAPE Average: {}\n'.format(np.mean(mape)))
    
    f.write('Test R2: {}\n'.format(r2))
    f.write('R2 Average: {}\n'.format(np.mean(r2)))
    
    f.write('Training Samples: {}\n'.format(X_train.shape[0]))
    f.write('Testing Samples: {}\n'.format(X_test.shape[0]))
    f.write('Input Features: {}\n'.format(X_train.shape[2]))
    f.write('Input Steps: {}\n'.format(X_train.shape[1]))
    f.write('Target Variables: {}\n'.format(num_target))
    f.write('Batch Size: {}\n'.format(256))
    f.write('Total Epochs: {}\n'.format(total_epochs))
    f.write('Training Time (s): {:.2f}\n'.format(total_time))
    

# 训练模型
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('ICLSTM Training and validation loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(f'{visualization_path}/ICLSTM_loss.png')
plt.show()

#画11个预测值的对比
num_targets = len(predict_columns)
for i in range(num_targets):
    plt.figure()  # 每个目标单独开一个图
    plt.plot(y_test[:, i], label='True ' + predict_columns[i])
    plt.plot(y_pred_real[:, i], label='Pred ' + predict_columns[i])
    plt.title('Comparison: ' + predict_columns[i])
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{visualization_path}/ICLSTM_comparison_{predict_columns[i]}.png')
    plt.show()