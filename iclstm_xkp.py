import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Activation, Dropout, Add, LSTM, GRU, RNN, LayerNormalization, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras import backend as K
from keras.optimizers import Adam,SGD
import tensorflow as tf
from keras import Model, regularizers, activations, initializers
import pickle
from sklearn.preprocessing import StandardScaler
from ICLSTM import MyICLSTMCell
# set the seed for reproducibility
tf.random.set_seed(42)
# 读取数据
data = pd.read_csv('simulation_output_short.csv')

# 配置
sequence_length = 40
predict_columns = ['Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 
                   'Z06_T', 'Z07_T', 'Z08_T', 'Bd_FracCh_Bat', 'Fa_Pw_Prod', 'Fa_E_All']

# 去除目标变量构建输入特征索引
input_columns = [col for col in data.columns if col not in predict_columns]
input_indices = [data.columns.get_loc(col) for col in input_columns]
target_indices = [data.columns.get_loc(col) for col in predict_columns]
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
                                                    random_state=123)
num_dims = 114
num_step = 40
num_target = 11
#归一化
scaler_X = preprocessing.StandardScaler().fit(X_train.reshape(-1, num_dims))
scaler_y = preprocessing.StandardScaler().fit(y_train.reshape(-1, num_target))
X_train = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
X_test = scaler_X.transform(X_test.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
y_train = scaler_y.transform(y_train.reshape(-1,num_target))
print('')


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
x = Dense(11, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())(x)
x = x[:,-1,:]
x = tf.reshape(x, (-1, num_target)) 
model = Model(input, x)

# 训练配置
# 1. 定义一个学习率衰减调度器
from tensorflow.keras.optimizers.schedules import ExponentialDecay
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,  # 初始学习率
    decay_steps=10000,           # 每隔多少 step 衰减一次
    decay_rate=0.1,             # 衰减系数(例如衰减到原来的96%)
    staircase=True               # True表示阶梯衰减，False表示连续指数衰减
)
optimizer = Adam(learning_rate=lr_schedule, clipnorm=None)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
history = model.fit(X_train, y_train, epochs=15, batch_size=256, validation_split=0.25, verbose=2)
model.save('iclstm.h5')
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
plt.savefig('visualization/ICLSTM_loss.png')

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
    plt.savefig('visualization/ICLSTM_comparison_{}.png'.format(predict_columns[i]))
    plt.show()