import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from keras.models import Model
from keras.layers import Dense, Input, Layer, Embedding, Dropout
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
model_path = os.path.join(visualization_path, 'transformer_480steps_without_time_feature.h5')

# -------------------------------------------------
# 2. 数据读取与基本配置
# -------------------------------------------------
data = pd.read_csv('simulation_output.csv')

sequence_length = 10
epochs = 500

# 输入和预测列
# input_columns_list = [
#     'Z01_T','Z02_T','Z03_T','Z04_T','Z05_T','Z06_T','Z07_T','Z08_T',
#     'Bd_FracCh_Bat','Fa_Pw_Prod','Fa_E_All','Fa_E_HVAC','Ext_T','Ext_Irr',
#     'PV_Gen_corrected','P1_T_Thermostat_sp_out','P2_T_Thermostat_sp_out',
#     'P3_T_Thermostat_sp_out','P4_T_Thermostat_sp_out','Bd_Pw_Bat_sp_out'
# ]
not_use_columns = ['Timestamp', 'Month', 'Day', 'Hour', 'Minute']
predict_columns = ['Fa_E_All','Fa_E_HVAC']

# 获取输入数据及预测数据
input_columns = [col for col in data.columns if col not in not_use_columns]
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

# 如果需要保持原有的“对称性”操作，进行拼接 X 与 -X（如原代码所示）
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

X_train = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1, num_step, num_dims)
X_test  = scaler_X.transform(X_test.reshape(-1, num_dims)).reshape(-1, num_step, num_dims)
y_train = scaler_y.transform(y_train.reshape(-1, num_target))

# -------------------------------------------------
# 6. 定义自定义层：ConvexMultiHeadAttention 和 TransformerBlock
# -------------------------------------------------

class ConvexMultiHeadAttention(Layer):
    """
    自定义的多头自注意力层，使用 Convex-r-Softmax 算子替换标准 softmax，
    并且生成 Q, K, V 时使用非负权重和非负对角矩阵（以对角向量实现）。
    """
    def __init__(self, d_model, num_heads, r=1.0, lam=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.head_dim = d_model // num_heads
        self.r = r      # 控制稀疏性的参数 r
        self.lam = lam  # 正则化超参数 λ

        # 共享的线性投影层 W_x，要求非负（即所有元素 >= 0）
        self.w_proj = Dense(d_model, kernel_constraint=tf.keras.constraints.NonNeg())

        # 对角缩放向量，分别用于生成 Q, K, V，初始均为 1，约束非负
        self.d_q = self.add_weight(shape=(d_model,), initializer='ones', trainable=True,
                                   constraint=tf.keras.constraints.NonNeg(), name="d_q")
        self.d_k = self.add_weight(shape=(d_model,), initializer='ones', trainable=True,
                                   constraint=tf.keras.constraints.NonNeg(), name="d_k")
        self.d_v = self.add_weight(shape=(d_model,), initializer='ones', trainable=True,
                                   constraint=tf.keras.constraints.NonNeg(), name="d_v")

    def call(self, x):
        # x 的 shape: (batch, seq_len, d_model)
        X_proj = self.w_proj(x)  # 非负线性变换，shape: (batch, seq_len, d_model)

        # 生成 Q, K, V：对 X_proj 分别乘以各自的对角向量（元素级乘法）
        Q = X_proj * self.d_q  # shape: (batch, seq_len, d_model)
        K = X_proj * self.d_k
        V = X_proj * self.d_v

        # 将 Q, K, V 分头：变换为 (batch, num_heads, seq_len, head_dim)
        def split_heads(x):
            batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[1]
            # reshape 成 (batch, seq_len, num_heads, head_dim)
            x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
            # transpose 成 (batch, num_heads, seq_len, head_dim)
            return tf.transpose(x, perm=[0, 2, 1, 3])
        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # 计算注意力 logits：Q 与 K 的点积，shape: (batch, num_heads, seq_len, seq_len)
        scores = tf.matmul(Q, K, transpose_b=True)

        # 定义 Convex-r-Softmax 算子：将 logits 转换为注意力权重
        def convex_r_softmax(z, r=1.0, lam=0.1, clip_min=-15.0, clip_max=15.0):
            """
            改进的 Convex-r-Softmax 实现。
            - 对 z 先做 clip，避免 exp(z - r) 溢出或变得过小。
            - 对分子做 ReLU，防止分子在加和后变为负数。
            - 对分母加上更大的 epsilon，防止出现除以 0。
            """
            # 1) 对 z 进行截断，防止出现极端大/小值导致 exp() 爆炸或下溢
            z = tf.clip_by_value(z, clip_min, clip_max)

            # 2) 按照原公式计算分子：exp(z - r) + λ*z
            numerator = tf.exp(z - r) + lam * z

            # 3) 让分子保持非负，以免分母求和后变成负值/过小
            numerator = tf.nn.relu(numerator)

            # 4) 求分母并加上较大的 epsilon
            denominator = tf.reduce_sum(numerator, axis=-1, keepdims=True) + 1e-9

            return numerator / denominator

        # 应用 Convex-r-Softmax 算子（沿最后一维，即 keys 维度）
        attn_weights = convex_r_softmax(scores)
        # attn_weights = tf.nn.softmax(scores, axis=-1)

        # 计算注意力输出：加权求和 V
        attn_output = tf.matmul(attn_weights, V)  # shape: (batch, num_heads, seq_len, head_dim)

        # 合并多头：将 (batch, num_heads, seq_len, head_dim) 转换回 (batch, seq_len, d_model)
        def combine_heads(x):
            batch_size = tf.shape(x)[0]
            # transpose 回 (batch, seq_len, num_heads, head_dim)
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            seq_len = tf.shape(x)[1]
            return tf.reshape(x, (batch_size, seq_len, self.d_model))
        output = combine_heads(attn_output)
        return output

class TransformerBlock(Layer):
    """
    Transformer Encoder Block：
      1. 自定义的多头自注意力层（ConvexMultiHeadAttention）
      2. 残差连接（不使用 LayerNorm，以保持整体凸性）
      3. 前向网络（Feed Forward Network），其中 Dense 层的权重受非负限制
      4. 残差连接
    """
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, r=1.0, lam=0.1, **kwargs):
        super().__init__(**kwargs)
        # 自定义多头自注意力层
        self.att = ConvexMultiHeadAttention(d_model=d_model, num_heads=num_heads, r=r, lam=lam)
        self.dropout1 = Dropout(rate)
        # 前向网络：两层 Dense，其中均使用非负权重约束；ReLU 保持凸性
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu", kernel_constraint=tf.keras.constraints.NonNeg()),
            Dense(d_model, kernel_constraint=tf.keras.constraints.NonNeg()),
        ])
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False, mask=None):
        # 1. 自注意力层
        attn_output = self.att(inputs)  # 输出 shape: (batch, seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # 残差连接（不做归一化，保持凸性）
        out1 = inputs + attn_output

        # 2. 前向网络层
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # 残差连接
        out2 = out1 + ffn_output
        return out2

class PositionalEmbedding(Layer):
    """
    位置嵌入层：对输入进行线性映射并加上可训练的位置嵌入。
    这里保持原有设计不变。
    """
    def __init__(self, sequence_length, d_model):
        super().__init__()
        # 将原始特征映射到 d_model 维度
        self.token_dense = Dense(d_model)
        # 可训练的位置嵌入
        self.pos_emb = Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        # 对输入做线性映射
        x = self.token_dense(x)
        # 生成位置序列：[0, 1, 2, ..., seq_len-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions)
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
    dropout_rate=0.1,
    r=1.0,    # Convex-r-Softmax 参数 r
    lam=0.1   # Convex-r-Softmax 参数 λ
):
    """
    构建 IC-EoT 模型：
      - input_shape: (num_step, num_dims)
      - num_target: 预测目标数量
      - 其他参数控制模型维度和注意力、前向网络的宽度
    """
    inputs = Input(shape=input_shape)  # shape: (batch, num_step, num_dims)

    # 位置嵌入（保持不变）
    x = PositionalEmbedding(sequence_length=input_shape[0], d_model=d_model)(inputs)

    # 堆叠 TransformerBlock（每层均使用自定义的 ConvexMultiHeadAttention 和无 LayerNorm 设计）
    for _ in range(num_layers):
        x = TransformerBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim,
                             rate=dropout_rate, r=r, lam=lam)(x)

    # 提取最后一个时间步的输出（类似于序列末尾的状态用于预测）
    x = x[:, -1, :]  # shape: (batch, d_model)

    # 输出层：使用线性激活，并可选择对输出层也加非负约束
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
    num_heads=1,      # 多头注意力头数
    ff_dim= 128,       # 前向网络宽度
    num_layers=1,     # TransformerBlock 层数
    dropout_rate=0.1, # Dropout 比例
    r=1.0,            # Convex-r-Softmax 参数 r
    lam=0.1           # Convex-r-Softmax 参数 λ
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
optimizer = Adam(learning_rate=5e-3) # 5e-3: 0.98
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
# model.save(model_path)

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
plt.figure()
plt.plot(epochs_range, loss, 'r', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Transformer Training and Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
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
