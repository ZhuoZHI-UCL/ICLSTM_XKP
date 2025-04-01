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

# 设置随机种子
tf.random.set_seed(42)

# 路径配置
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
visualization_path = 'visualization/lstm_480steps_without_time_feature/'
os.makedirs(visualization_path, exist_ok=True)
test_result_path = os.path.join(visualization_path, 'test_result.txt')
model_path = os.path.join(visualization_path, 'lstm_480steps_without_time_feature.h5')

# 数据读取
data = pd.read_csv('simulation_output.csv')

sequence_length = 480
epochs = 200

input = ['Z01_T','Z02_T','Z03_T','Z04_T','Z05_T','Z06_T','Z07_T','Z08_T','Bd_FracCh_Bat','Fa_Pw_Prod','Fa_E_All','Fa_E_HVAC','Ext_T','Ext_Irr','PV_Gen_corrected','P1_T_Thermostat_sp_out','P2_T_Thermostat_sp_out','P3_T_Thermostat_sp_out','P4_T_Thermostat_sp_out','Bd_Pw_Bat_sp_out']
predict_columns = ['Z01_T','Z02_T','Z03_T','Z04_T','Z05_T','Z06_T','Z07_T','Z08_T','Bd_FracCh_Bat','Fa_Pw_Prod','Fa_E_All','Fa_E_HVAC','Ext_T','Ext_Irr','PV_Gen_corrected']

# 获取列索引
input_columns = [col for col in data.columns if col in input]
input_indices = [data.columns.get_loc(col) for col in input_columns]
target_indices = [data.columns.get_loc(col) for col in predict_columns]

input_data = data[input_columns].values
target_data = data[predict_columns].values

# 构建序列数据
X, Y = [], []
for i in range(len(data) - sequence_length):
    seq_input = input_data[i : i + sequence_length]
    seq_target = target_data[i + sequence_length]
    X.append(seq_input)
    Y.append(seq_target)

X = np.array(X)
X = np.concatenate((X, -X), axis=2)
Y = np.array(Y)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123, shuffle=False)

num_dims = X_train.shape[2]
num_step = sequence_length
num_target = len(predict_columns)
print(f'we use {num_dims} features, {num_step} steps, {num_target} targets')

# 标准化
scaler_X = preprocessing.StandardScaler().fit(X_train.reshape(-1, num_dims))
scaler_y = preprocessing.StandardScaler().fit(y_train.reshape(-1, num_target))
X_train = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
X_test = scaler_X.transform(X_test.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
y_train = scaler_y.transform(y_train.reshape(-1,num_target))

# 模型定义：标准LSTM替代ICLSTM
inputs = Input(shape=(num_step, num_dims))
x_skip = inputs
x = LSTM(units=128, return_sequences=True)(inputs)
x = Dense(num_dims, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = Add()([x, x_skip])

x = LSTM(units=128, return_sequences=True)(x)
x = Dense(num_dims, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = Add()([x, x_skip])

x = Dense(num_target, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = x[:, -1, :]  # 取最后一个时间步
x = tf.reshape(x, (-1, num_target))
model = Model(inputs, x)

# 训练配置
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

# 模型训练
start_time = time.time()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=256, validation_split=0.25, verbose=2, callbacks=[early_stopping])
end_time = time.time()
total_time = end_time - start_time
total_epochs = len(history.history['loss'])

model.save(model_path)

# 测试评估
y_pred_scaled = model.predict(X_test)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
mse = mean_squared_error(y_test, y_pred_real, multioutput='raw_values')
mape = mean_absolute_percentage_error(y_test, y_pred_real, multioutput='raw_values')
r2 = r2_score(y_test, y_pred_real, multioutput='raw_values')

print("Test MSE:", mse)
print("Test MAPE:", mape)
print("Test R2:", r2)

# 保存结果
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

# 可视化损失曲线
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(loss) + 1)
plt.plot(epochs_range, loss, 'r', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('LSTM Training and validation loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(f'{visualization_path}/LSTM_loss.png')

# 逐个预测结果绘图
num_targets = len(predict_columns)
for i in range(num_targets):
    plt.figure()
    plt.plot(y_test[:, i], label='True ' + predict_columns[i])
    plt.plot(y_pred_real[:, i], label='Pred ' + predict_columns[i])
    plt.title('Comparison: ' + predict_columns[i])
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{visualization_path}/LSTM_comparison_{predict_columns[i]}.png')
    plt.show()
