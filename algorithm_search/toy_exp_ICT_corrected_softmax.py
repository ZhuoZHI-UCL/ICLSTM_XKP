import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras import Model
from keras.layers import Dense, Input, Layer, Embedding, Dropout
from keras.optimizers import Adam
from tensorflow.keras.constraints import NonNeg
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
import random

# -------------------------------------------------
# 定义 Convex-r-Softmax 函数（原 sparse_convex_softmax，只改变量名称）
# -------------------------------------------------
def convex_r_softmax(z, r, tau):
    """
    Convex-r-Softmax 函数
    - z: 输入张量，形状为 (..., n)
    - r: 阈值参数，控制稀疏性（可标量或与 z 同形状）
    - tau: 温度参数，需 > 0
    """
    # 确保 tau > 0
    tau = tf.maximum(tau, 1e-9)

    # 平移并缩放输入
    shifted_z = (z - r) / tau

    # 稳定化计算：减去最大值防止溢出
    max_z = tf.reduce_max(shifted_z, axis=-1, keepdims=True)
    exp_z = tf.exp(shifted_z - max_z)

    # 计算归一化分母
    sum_exp = tf.reduce_sum(exp_z, axis=-1, keepdims=True) + 1e-9

    # 输出概率分布
    return exp_z / sum_exp

# -------------------------------------------------
# 设置随机种子以保证结果可复现
# -------------------------------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# -------------------------------------------------
# 定义全局配置对象，统一管理超参数 r 和 tau
# -------------------------------------------------
class HyperParams:
    def __init__(self, r=0.3, tau=1.0):
        self.r = r
        self.tau = tau

# 在此只需调整一次配置，即可应用到整个模型中
config = HyperParams(r=1, tau=100.0)

# -------------------------------------------------
# 1. 生成 toy 数据集
# -------------------------------------------------
ll = np.linspace(-1, 1, 60)
xx, yy = np.meshgrid(ll, ll)

# toy 函数：z = -cos(4*(x^2 + y^2))
# zz = -np.cos(4 * (xx**2 + yy**2))
# 可选的其他函数：
# zz = np.fmax(np.fmin(xx**2 + yy**2, (2*xx-1)**2  + (2*yy-1)**2 - 2),
#             -(2*xx+1)**2  - (2*yy+1)**2 + 4)
zz = xx**2 * (4 - 2.1 * xx**2 + xx**4 / 3) - 4 * yy**2 * (1 - yy**2) + xx * yy

# 将 (xx, yy) 打平并叠到第三维
inps = np.stack([xx.reshape(-1,1), yy.reshape(-1,1)], axis=-1)  # (3600, 1, 2)
inps = inps.repeat(5, axis=1)  # (3600, 5, 2)

# 在特征维度上拼接正负两份，得到 (3600, 5, 4)
inps_ = np.concatenate([inps, -inps], axis=-1)

# 目标值 reshape 成 (3600, 1, 1)
targs = zz.reshape(-1, 1, 1)

# 划分 train / test 数据集
train_inps, test_inps, train_targs, test_targs = train_test_split(inps_, targs, test_size=0.3)

# -------------------------------------------------
# 2. 定义自定义的多头注意力层（ConvexMultiHeadAttention）
# -------------------------------------------------
class ConvexMultiHeadAttention(Layer):
    """
    使用 Convex-r-Softmax 替换标准 softmax，并在生成 Q, K, V 时采用非负权重及对角向量。
    超参数 r 和 tau 均由全局配置对象统一管理。
    """
    def __init__(self, d_model, num_heads, config, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.head_dim = d_model // num_heads

        # 使用配置对象中的超参数
        self.r = config.r
        self.tau = config.tau

        # 非负线性投影
        self.w_proj = Dense(d_model, kernel_constraint=NonNeg())

        # 可训练的非负对角向量，初始为 1
        self.d_q = self.add_weight(shape=(d_model,),
                                   initializer='ones',
                                   trainable=True,
                                   constraint=NonNeg(),
                                   name="d_q")
        self.d_k = self.add_weight(shape=(d_model,),
                                   initializer='ones',
                                   trainable=True,
                                   constraint=NonNeg(),
                                   name="d_k")
        self.d_v = self.add_weight(shape=(d_model,),
                                   initializer='ones',
                                   trainable=True,
                                   constraint=NonNeg(),
                                   name="d_v")

    def call(self, x):
        # x shape: (batch, seq_len, input_dim)
        X_proj = self.w_proj(x)  # 映射到 d_model: (batch, seq_len, d_model)

        # 元素乘以对角向量得到 Q, K, V
        Q = X_proj * self.d_q
        K = X_proj * self.d_k
        V = X_proj * self.d_v

        # 将 Q, K, V 按头拆分
        def split_heads(tensor):
            batch_size = tf.shape(tensor)[0]
            seq_len = tf.shape(tensor)[1]
            tensor = tf.reshape(tensor, (batch_size, seq_len, self.num_heads, self.head_dim))
            return tf.transpose(tensor, perm=[0, 2, 1, 3])

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # 计算注意力分数
        scores = tf.matmul(Q, K, transpose_b=True)

        # 使用 Convex-r-Softmax 替换原有的 softmax 实现
        attn_weights = convex_r_softmax(scores, self.r, self.tau)
        attn_output = tf.matmul(attn_weights, V)

        # 合并多头
        def combine_heads(tensor):
            batch_size = tf.shape(tensor)[0]
            tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
            return tf.reshape(tensor, (batch_size, -1, self.d_model))

        output = combine_heads(attn_output)
        return output

# -------------------------------------------------
# 3. 定义 TransformerBlock
# -------------------------------------------------
class TransformerBlock(Layer):
    """
    包含 ConvexMultiHeadAttention、前向网络及残差连接。
    超参数 r 和 tau 均通过配置对象传递给内部注意力层。
    """
    def __init__(self, d_model, num_heads, ff_dim, config, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = ConvexMultiHeadAttention(d_model, num_heads, config)
        self.dropout1 = Dropout(rate)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu", kernel_constraint=NonNeg()),
            Dense(d_model, kernel_constraint=NonNeg())
        ])
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output  # 残差连接

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output  # 残差连接
        return out2

# -------------------------------------------------
# 4. 定义 PositionalEmbedding（可训练的位置编码）
# -------------------------------------------------
class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        # 先将输入通过全连接映射到 d_model，不再限制权重非负
        self.token_dense = Dense(d_model)
        # 可训练的位置编码
        self.pos_emb = Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.token_dense(x)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions)
        return x + pos_embeddings

# -------------------------------------------------
# 5. 搭建 IC-Transformer 模型 (返回所有时间步)
# -------------------------------------------------
def build_transformer_model(
    input_shape,       # (seq_len, num_feature)
    d_model=128,
    num_heads=1,
    ff_dim=128,
    num_layers=1,
    dropout_rate=0.1,
    config=None
):
    inputs = Input(shape=input_shape)
    # 使用可训练位置编码
    x = PositionalEmbedding(sequence_length=input_shape[0], d_model=d_model)(inputs)

    # 堆叠 TransformerBlock，每个 Block 均共享全局配置的超参数
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim, config, rate=dropout_rate)(x)

    # 对每个时间步均进行预测
    outputs = Dense(1, activation='linear', kernel_constraint=NonNeg())(x)
    model = Model(inputs, outputs)
    return model

# -------------------------------------------------
# 6. 构建并训练 Transformer 模型
# -------------------------------------------------
transformer_model = build_transformer_model(
    input_shape=(train_inps.shape[1], train_inps.shape[2]),
    d_model=128,
    num_heads=1,
    ff_dim=128,
    num_layers=1,
    dropout_rate=0.1,
    config=config
)

optimizer = Adam(learning_rate=1e-3)
transformer_model.compile(optimizer=optimizer,
                          loss='mean_squared_error',
                          metrics=[tf.keras.metrics.MeanSquaredError()])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = transformer_model.fit(
    train_inps, train_targs,
    epochs=2000,
    batch_size=64,
    validation_split=0.25,
    callbacks=[early_stopping],
    verbose=2
)

transformer_model.summary()

# -------------------------------------------------
# 7. 训练过程可视化
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(3,3))
ax.plot(history.history['loss'], 'r', label='Train loss')
ax.plot(history.history['val_loss'], 'b', label='Valid loss')
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_yscale('log')
ax.legend()
ax.plot()
plt.show()

# -------------------------------------------------
# 8. 分别查看每个时间步的预测结果，并与真值对比
# -------------------------------------------------
# 整体输入: (3600, 5, 4)
# 预测输出: (3600, 5, 1)
predictions = transformer_model.predict(inps_.astype(np.float32))  # (3600, 5, 1)

# xx, yy, zz 用于可视化 => 60 x 60
# 依次查看 time_step = 0,1,2,3,4 的预测
for t in range(5):
    preds_t = predictions[:, t, 0].reshape((60, 60))

    plt.close('all')
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xx, yy, zz,    alpha=0.5, label='true', color='orange')
    ax.plot_wireframe(xx, yy, preds_t, alpha=0.8, label=f'pred (t={t+1})', color='blue')
    ax.legend()
    plt.savefig(f'./pred_t{t+1}.png', dpi=300, bbox_inches='tight')
    plt.show()
