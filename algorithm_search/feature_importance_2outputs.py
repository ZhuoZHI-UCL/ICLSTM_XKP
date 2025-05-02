import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Add
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
import time
import os
import random

# ============== 1. 设置随机种子 & 目录配置 ==============
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 如果不用指定GPU可注释掉
visualization_path = 'visualization/LSTM_10steps_without_time_feature_2outputs/'
os.makedirs(visualization_path, exist_ok=True)
test_result_path = os.path.join(visualization_path, 'test_result.txt')
model_path = os.path.join(visualization_path, 'LSTM_10steps_without_time_feature_2outputs.h5')

# ============== 2. 读取数据 & 相关配置 ==============
data = pd.read_csv('simulation_output.csv')

sequence_length = 10
epochs = 200

# 不使用的列
not_use_columns = ['Timestamp', 'Month', 'Day', 'Hour', 'Minute']
# 需要预测的目标列
predict_columns = ['Fa_E_All','Fa_E_HVAC']

# 输入特征列：如果需要把“目标列”也视作输入，请去掉下面的 (col not in predict_columns)
input_columns = [
    col for col in data.columns 
    if (col not in not_use_columns and col not in predict_columns)
]

# 取出输入、目标的 numpy 数组
input_data = data[input_columns].values
target_data = data[predict_columns].values

# ============== 3. 构建时序数据序列 X, Y ==============
X, Y = [], []
for i in range(len(data) - sequence_length):
    seq_input = input_data[i : i + sequence_length]   # 过去10步的数据
    seq_target = target_data[i + sequence_length]     # 预测第10步之后那个时刻的目标
    X.append(seq_input)
    Y.append(seq_target)

X = np.array(X)  # (samples, timesteps, features)
Y = np.array(Y)

# 在最后一维拼接 -X，变成 2 倍特征数
X = np.concatenate((X, -X), axis=2)

# ============== 4. 训练集、测试集划分 ==============
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.3, 
    random_state=123, 
    shuffle=False
)

num_dims = X_train.shape[2]
num_step = sequence_length
num_target = len(predict_columns)

print(f'we use {num_dims} features, {num_step} steps, {num_target} targets')

# ============== 5. 标准化 ==============
scaler_X = preprocessing.StandardScaler().fit(X_train.reshape(-1, num_dims))
scaler_y = preprocessing.StandardScaler().fit(y_train.reshape(-1, num_target))

X_train = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
X_test  = scaler_X.transform(X_test.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
y_train = scaler_y.transform(y_train.reshape(-1,num_target))

# ============== 6. 定义并训练模型 ==============
inputs = Input(shape=(num_step, num_dims))
x_skip = inputs

x = LSTM(units=128, return_sequences=True)(inputs)
x = Dense(num_dims, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = Add()([x, x_skip])

x = LSTM(units=128, return_sequences=True)(x)
x = Dense(num_dims, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = Add()([x, x_skip])

x = Dense(num_target, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = x[:, -1, :]  # 只取最后一个时间步
x = tf.reshape(x, (-1, num_target))

model = Model(inputs, x)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

start_time = time.time()
history = model.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=256, 
    validation_split=0.25, 
    verbose=2, 
    callbacks=[early_stopping]
)
end_time = time.time()
total_time = end_time - start_time
total_epochs = len(history.history['loss'])

model.save(model_path)

# ============== 7. 测试评估 ==============
y_pred_scaled = model.predict(X_test)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test, y_pred_real, multioutput='raw_values')
mape = mean_absolute_percentage_error(y_test, y_pred_real, multioutput='raw_values')
r2 = r2_score(y_test, y_pred_real, multioutput='raw_values')

print("Test MSE:", mse)
print("Test MAPE:", mape)
print("Test R2:", r2)

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

# ============== 8. 训练 & 预测结果可视化 ==============
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs_range, loss, 'r', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('LSTM Training and Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'{visualization_path}/LSTM_loss.png')
plt.show()

for i in range(num_target):
    plt.figure()
    plt.plot(y_test[:, i], label='True ' + predict_columns[i])
    plt.plot(y_pred_real[:, i], label='Pred ' + predict_columns[i])
    plt.title('Comparison: ' + predict_columns[i])
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{visualization_path}/LSTM_comparison_{predict_columns[i]}.png')
    plt.show()

# ============== 9. 使用 R² 做置乱重要性 (Permutation Importance) ==============

# -- 9.1 定义“负的 R²”函数，让它越低越好，保持与 MSE 相同的差值公式
def neg_r2_score(y_true, y_pred):
    """
    返回 -r2_score(y_true, y_pred),
    这样越大(负值更大->实际值更小)表示模型性能越差，可与 MSE 做同样的差值操作。
    """
    return -r2_score(y_true, y_pred, multioutput='uniform_average')

# -- 9.2 定义 permutation_importance 函数
def permutation_importance(
    model, 
    X_test, 
    y_test, 
    feature_names, 
    metric_func=neg_r2_score,  # 这里默认用负的 R²
    batch_size=256,
    random_seed=42
):
    """
    对 LSTM (或任意模型) 做基于置乱(permutation)的特征重要性分析。
    
    参数:
    -------
    model : keras.Model
        已经训练好的 Keras 模型
    X_test : np.array, shape = (n_samples, n_timesteps, n_features)
        测试集输入特征
    y_test : np.array, shape = (n_samples, n_outputs)
        测试集真实值
    feature_names : list of str
        每个特征对应的名称，长度必须与 n_features 一致
    metric_func : callable
        度量误差的函数, 输入 (y_true, y_pred) -> 输出标量. 
        (此处用 neg_r2_score, 值越低表示性能越好)
    batch_size : int
        预测时的 batch_size
    random_seed : int
        方便固定随机数种子
    
    返回:
    -------
    baseline_score : float
        不置乱时模型的“负的 R²”分数
    importance_df : pd.DataFrame
        含 [feature_name, importance] 的 DataFrame.
        importance = permuted_score - baseline_score (对 neg_r2)
        => baseline_r2 - permuted_r2
        数值越大, 说明置乱导致 R² 降得越多, 特征越重要.
    """
    np.random.seed(random_seed)

    # 1) baseline：原始数据上的“负的 R²”
    y_pred = model.predict(X_test, batch_size=batch_size)
    baseline_score = metric_func(y_test, y_pred)  # 负的 R²

    n_features = X_test.shape[2]
    importances = []
    
    # 2) 对每个特征进行置乱
    for f_idx in range(n_features):
        X_permuted = X_test.copy()
        
        # 将第 f_idx 个特征在所有样本、所有时间步上打乱
        shuffled_feature = X_permuted[:,:,f_idx].reshape(-1).copy()
        np.random.shuffle(shuffled_feature)
        X_permuted[:,:,f_idx] = shuffled_feature.reshape(X_permuted[:,:,f_idx].shape)
        
        # 3) 重新预测并计算“负的 R²”
        y_permuted_pred = model.predict(X_permuted, batch_size=batch_size)
        permuted_score = metric_func(y_test, y_permuted_pred)
        
        # importance = permuted_score - baseline_score
        # 对于 neg_r2_score 来说, importance = -(R²_permuted) - (-(R²_baseline))
        #                              = R²_baseline - R²_permuted
        importances.append(permuted_score - baseline_score)
    
    # 4) 整理输出
    importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    return baseline_score, importance_df

# -- 9.3 构造特征名列表（因为做了拼接 X, -X）
original_feature_names = input_columns
neg_feature_names = [f"neg_{col}" for col in original_feature_names]
all_feature_names = original_feature_names + neg_feature_names

assert len(all_feature_names) == X_test.shape[2], "特征名数量与 X_test.shape[2] 不匹配!"

# -- 9.4 计算置乱重要性
baseline_neg_r2, importance_df = permutation_importance(
    model,
    X_test,
    y_test,
    all_feature_names,
    metric_func=neg_r2_score,  # 这里用负的R²
    batch_size=256,
    random_seed=42
)

# baseline_neg_r2 = -baseline_r2
baseline_r2 = -baseline_neg_r2
print("Baseline R² (no shuffle):", baseline_r2)
print("\nPermutation Importance 排序 (数值= baseline_r2 - permuted_r2)：")
print(importance_df)

# 如果需要将结果写入CSV文件:
importance_df.to_csv(os.path.join(visualization_path, 'feature_importance_r2.csv'), index=False)
