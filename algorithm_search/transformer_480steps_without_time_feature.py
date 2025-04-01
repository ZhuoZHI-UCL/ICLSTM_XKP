import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from keras.models import Model
from keras.layers import Dense, Input, Layer, Embedding, Dropout, LayerNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
import time
import os
import random
# -------------------------------------------------
# 1. 设置随机种子和路径
# -------------------------------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
visualization_path = 'visualization/transformer_480steps_without_time_feature/'
os.makedirs(visualization_path, exist_ok=True)
test_result_path = os.path.join(visualization_path, 'test_result.txt')
model_path = os.path.join(visualization_path, 'transformer_480steps_without_time_feature.h5')  # 这里名字可以不改

# -------------------------------------------------
# 2. 数据读取与基本配置
# -------------------------------------------------
data = pd.read_csv('simulation_output.csv')

sequence_length = 480
epochs = 200

# 输入和预测列
input = [
    'Z01_T','Z02_T','Z03_T','Z04_T','Z05_T','Z06_T','Z07_T','Z08_T',
    'Bd_FracCh_Bat','Fa_Pw_Prod','Fa_E_All','Fa_E_HVAC','Ext_T','Ext_Irr',
    'PV_Gen_corrected','P1_T_Thermostat_sp_out','P2_T_Thermostat_sp_out',
    'P3_T_Thermostat_sp_out','P4_T_Thermostat_sp_out','Bd_Pw_Bat_sp_out'
]
predict_columns = [
    'Z01_T','Z02_T','Z03_T','Z04_T','Z05_T','Z06_T','Z07_T','Z08_T',
    'Bd_FracCh_Bat','Fa_Pw_Prod','Fa_E_All','Fa_E_HVAC','Ext_T','Ext_Irr','PV_Gen_corrected'
]

# 获取列索引
input_columns = [col for col in data.columns if col in input]
input_indices = [data.columns.get_loc(col) for col in input_columns]
target_indices = [data.columns.get_loc(col) for col in predict_columns]

input_data = data[input_columns].values
target_data = data[predict_columns].values

# -------------------------------------------------
# 3. 构建序列数据
# -------------------------------------------------
X, Y = [], []
for i in range(len(data) - sequence_length):
    seq_input = input_data[i : i + sequence_length]
    seq_target = target_data[i + sequence_length]
    X.append(seq_input)
    Y.append(seq_target)

X = np.array(X)
Y = np.array(Y)

# 这里原代码中有 “X = np.concatenate((X, -X), axis=2)” 的操作，
# 如果你仍需要这个操作，请取消下面注释；否则和原逻辑保持相同。
X = np.concatenate((X, -X), axis=2)

# -------------------------------------------------
# 4. 划分训练集和测试集
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=123, shuffle=False
)

num_dims = X_train.shape[2]
num_step = sequence_length
num_target = len(predict_columns)
print(f'we use {num_dims} features, {num_step} steps, {num_target} targets')

# -------------------------------------------------
# 5. 数据标准化
# -------------------------------------------------
scaler_X = preprocessing.StandardScaler().fit(X_train.reshape(-1, num_dims))
scaler_y = preprocessing.StandardScaler().fit(y_train.reshape(-1, num_target))

X_train = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
X_test  = scaler_X.transform(X_test.reshape(-1, num_dims)).reshape(-1,num_step,num_dims)
y_train = scaler_y.transform(y_train.reshape(-1,num_target))

# -------------------------------------------------
# 6. 定义简易 Transformer 所需层
# -------------------------------------------------
class TransformerBlock(Layer):
    """
    一个基础的 Transformer Encoder Block:
    1) 多头自注意力(MultiHeadAttention)
    2) 残差 + LayerNorm
    3) 前向网络(Feed Forward)
    4) 残差 + LayerNorm
    """
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False, mask=None):
        # 多头自注意力
        attn_output = self.att(inputs, inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # 残差连接 + LayerNorm

        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)     # 残差连接 + LayerNorm
        return out2


class PositionalEmbedding(Layer):
    """
    将输入映射到 d_model 的维度，并加上可训练的位置嵌入。
    对于时间序列，可以采用正弦/余弦位置编码或其他方案。
    这里为了方便，使用了一个可训练 Embedding 来表示序列位置。
    """
    def __init__(self, sequence_length, d_model):
        super().__init__()
        # 将原始特征映射到 d_model 维度
        self.token_dense = Dense(d_model)
        # 可训练的位置向量
        self.pos_emb = Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        # 对输入做一个线性映射
        x = self.token_dense(x)
        # 生成位置序列 [0, 1, 2, ..., seq_len-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        # 嵌入位置
        pos_embeddings = self.pos_emb(positions)
        # 相加得到最终的嵌入表示
        return x + pos_embeddings

# -------------------------------------------------
# 7. 搭建 Transformer 模型
# -------------------------------------------------
def build_transformer_model(
    input_shape,
    num_target,
    d_model=128,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    dropout_rate=0.1
):
    """
    构建一个简易的 Transformer Encoder 模型。
    - input_shape: (num_step, num_dims)
    - num_target:  需要预测的目标数量
    """
    inputs = Input(shape=input_shape)  # (batch, num_step, num_dims)

    # 位置嵌入
    x = PositionalEmbedding(sequence_length=input_shape[0], d_model=d_model)(inputs)

    # 多层 TransformerBlock
    for _ in range(num_layers):
        x = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            rate=dropout_rate
        )(x)

    # 最后一个时间步的输出
    # 与 LSTM 部分保持一致，取序列的最后一步进行预测
    x = x[:, -1, :]  # shape: (batch, d_model)

    # 输出层
    # 如果仍想保持 NonNeg 约束，可以如下写
    outputs = Dense(num_target, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())(x)

    model = Model(inputs, outputs)
    return model

# -------------------------------------------------
# 8. 构建并编译模型
# -------------------------------------------------
model = build_transformer_model(
    input_shape=(num_step, num_dims),
    num_target=num_target,
    d_model=128,      # 隐藏维度
    num_heads=4,      # 多头注意力的头数
    ff_dim=256,       # 前向网络的宽度
    num_layers=2,     # 堆叠几个 TransformerBlock
    dropout_rate=0.1  # Dropout 比例
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

# -------------------------------------------------
# 9. 模型训练
# -------------------------------------------------
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

# 保存模型
model.save(model_path)

# -------------------------------------------------
# 10. 测试集评估
# -------------------------------------------------
y_pred_scaled = model.predict(X_test)
# 将预测结果从标准化空间逆变换回真实值
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test, y_pred_real, multioutput='raw_values')
mape = mean_absolute_percentage_error(y_test, y_pred_real, multioutput='raw_values')
r2 = r2_score(y_test, y_pred_real, multioutput='raw_values')

print("Test MSE:", mse)
print("Test MAPE:", mape)
print("Test R2:", r2)

# -------------------------------------------------
# 11. 保存结果到文件
# -------------------------------------------------
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

# -------------------------------------------------
# 12. 损失曲线可视化
# -------------------------------------------------
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(loss) + 1)
plt.plot(epochs_range, loss, 'r', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Transformer Training and validation loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(f'{visualization_path}/Transformer_loss.png')
plt.show()

# -------------------------------------------------
# 13. 逐个预测结果绘图
# -------------------------------------------------
num_targets = len(predict_columns)
for i in range(num_targets):
    plt.figure()
    plt.plot(y_test[:, i], label='True ' + predict_columns[i])
    plt.plot(y_pred_real[:, i], label='Pred ' + predict_columns[i])
    plt.title('Comparison: ' + predict_columns[i])
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{visualization_path}/Transformer_comparison_{predict_columns[i]}.png')
    plt.show()
