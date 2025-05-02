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
# set the seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# -------------------------------------------------
# 1. 生成 toy 数据集
# -------------------------------------------------
ll = np.linspace(-1, 1, 60)
xx, yy = np.meshgrid(ll, ll)

# toy 函数：z = -cos(4*(x^2 + y^2))
zz = -np.cos(4 * (xx**2 + yy**2))

# 将 (xx, yy) 打平并叠到第三维
# 初始形状：(3600, 2)，然后人为构造 5 个时间步（repeat(axis=1)）
inps = np.stack([xx.reshape(-1,1), yy.reshape(-1,1)], axis=-1)  # (3600, 1, 2)
inps = inps.repeat(5, axis=1)  # (3600, 5, 2)

# 在特征维度上再拼接正负两份 (axis=-1)，得到 (3600, 5, 4)
inps_ = np.concatenate([inps, -inps], axis=-1)

# 目标值 zz reshape 成 (3600, 1, 1)
targs = zz.reshape(-1, 1, 1)

# 划分 train / test
train_inps, test_inps, train_targs, test_targs = train_test_split(inps_, targs, test_size=0.3)


# -------------------------------------------------
# 2. 定义自定义的多头注意力层（ConvexMultiHeadAttention）
# -------------------------------------------------
class ConvexMultiHeadAttention(Layer):
    """
    自定义的多头自注意力层，使用 Convex-r-Softmax 替换标准 softmax，
    并且在生成 Q, K, V 时使用非负权重和非负对角向量。
    """
    def __init__(self, d_model, num_heads, r=1.0, lam=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.head_dim = d_model // num_heads
        self.r = r      # 控制稀疏性的参数 r
        self.lam = lam  # 正则化超参数 λ

        # 非负线性投影 W_proj
        self.w_proj = Dense(d_model, kernel_constraint=NonNeg())

        # 三个可训练的对角向量 d_q, d_k, d_v（元素非负），初始为 1
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
        # x shape: (batch, seq_len, d_model_in)；但经过 w_proj 映射后变成 d_model
        X_proj = self.w_proj(x)  # (batch, seq_len, d_model)

        # 分别元素乘以 d_q, d_k, d_v 得到 Q, K, V
        Q = X_proj * self.d_q
        K = X_proj * self.d_k
        V = X_proj * self.d_v

        # split 多头
        def split_heads(tensor):
            batch_size = tf.shape(tensor)[0]
            seq_len = tf.shape(tensor)[1]
            # 先 reshape => (batch, seq_len, num_heads, head_dim)
            tensor = tf.reshape(tensor, (batch_size, seq_len, self.num_heads, self.head_dim))
            # 再 transpose => (batch, num_heads, seq_len, head_dim)
            return tf.transpose(tensor, perm=[0, 2, 1, 3])

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # 计算注意力 scores = QK^T
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, num_heads, seq_len, seq_len)

        # Convex-r-Softmax
        def convex_r_softmax(z, r=1.0, lam=0.1, clip_min=-15.0, clip_max=15.0):
            z = tf.clip_by_value(z, clip_min, clip_max)
            numerator = tf.exp(z - r) + lam * z
            numerator = tf.nn.relu(numerator)
            denominator = tf.reduce_sum(numerator, axis=-1, keepdims=True) + 1e-9
            return numerator / denominator

        attn_weights = convex_r_softmax(scores, self.r, self.lam)

        # 加权求和 V => 注意力输出
        attn_output = tf.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)

        # 合并多头
        def combine_heads(tensor):
            batch_size = tf.shape(tensor)[0]
            # transpose => (batch, seq_len, num_heads, head_dim)
            tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
            seq_len = tf.shape(tensor)[1]
            # reshape => (batch, seq_len, d_model)
            return tf.reshape(tensor, (batch_size, seq_len, self.d_model))

        output = combine_heads(attn_output)
        return output


# -------------------------------------------------
# 3. 定义一个 TransformerBlock
# -------------------------------------------------
class TransformerBlock(Layer):
    """
    包含：
      1) ConvexMultiHeadAttention
      2) 残差连接
      3) 两层前向网络 (FFN)，均为非负权重 + ReLU
      4) 残差连接
    不使用 LayerNorm。
    """
    def __init__(self, d_model, num_heads, ff_dim,
                 rate=0.1, r=1.0, lam=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = ConvexMultiHeadAttention(d_model, num_heads, r=r, lam=lam)
        self.dropout1 = Dropout(rate)

        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu", kernel_constraint=NonNeg()),
            Dense(d_model, kernel_constraint=NonNeg())
        ])
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)             # (batch, seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output                # 残差连接

        ffn_output = self.ffn(out1)                # (batch, seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output                   # 残差连接
        return out2


# -------------------------------------------------
# 4. 定义一个 PositionalEmbedding（可训练的位置编码）
# -------------------------------------------------
class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        # 先将原始输入做一个 Dense 到 d_model
        self.token_dense = Dense(d_model, kernel_constraint=NonNeg())
        # 可训练的位置向量
        self.pos_emb = Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, x):
        # x shape: (batch, seq_len, num_feature)
        seq_len = tf.shape(x)[1]
        x = self.token_dense(x)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions)  # (seq_len, d_model)
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
    r=1.0,             # Convex-r-Softmax 中的 r
    lam=0.1            # Convex-r-Softmax 中的 λ
):
    inputs = Input(shape=input_shape)  # (batch, seq_len, num_feature)

    # 可训练位置编码
    x = PositionalEmbedding(sequence_length=input_shape[0], d_model=d_model)(inputs)

    # 堆叠若干 TransformerBlock
    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim,
                             rate=dropout_rate, r=r, lam=lam)(x)

    # 最终输出：对每个时间步都做预测 (Dense => 1)
    # 若只想用最后一个时间步，可以改成 x = x[:, -1, :] 再 Dense(1)。
    outputs = Dense(1, activation='linear', kernel_constraint=NonNeg())(x)

    model = Model(inputs, outputs)
    return model


# -------------------------------------------------
# 6. 构建并训练 Transformer 模型
# -------------------------------------------------
# input_shape=(5,4)
transformer_model = build_transformer_model(
    input_shape=(train_inps.shape[1], train_inps.shape[2]),
    d_model=128,
    num_heads=1,
    ff_dim=128,
    num_layers=1,
    dropout_rate=0.1,
    r=1.0,
    lam=0.1
)

optimizer = Adam(learning_rate=5e-3)
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
    plt.show()

    # 如果需要保存某一幅图：
    fig.savefig("ictransformer_cosine_t{}.pdf".format(t))
