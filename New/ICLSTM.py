
import numpy as np
import matplotlib.pyplot as plt
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


class MyICLSTMCell(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [self.units, self.units]
        super(MyICLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wi = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                                      name='Wi',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      trainable=True)
        self.Ui = self.add_weight(shape=(self.units, self.units),
                                      initializer=initializers.Orthogonal(0.1),
                                      name='Ui',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      trainable=True)
        self.DWi = self.add_weight(shape=(self.units,),
                                      initializer=initializers.RandomUniform(minval=0, maxval=1),
                                      name='DWi',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      trainable=True)
        self.bi = self.add_weight(shape=(self.units,), initializer='zeros', name='bi', trainable=True)

        self.DWf = self.add_weight(shape=(self.units,),
                                      initializer=initializers.RandomUniform(minval=0, maxval=1),
                                      name='DWf',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      trainable=True)
        self.bf = self.add_weight(shape=(self.units,), initializer='zeros', name='bf', trainable=True)

        self.DWo = self.add_weight(shape=(self.units,),
                                      initializer=initializers.RandomUniform(minval=0, maxval=1),
                                      name='DWo',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      trainable=True)
        self.bo = self.add_weight(shape=(self.units,), initializer='zeros', name='bo', trainable=True)

        self.DWc = self.add_weight(shape=(self.units,),
                                      initializer=initializers.RandomUniform(minval=0, maxval=1),
                                      name='DWc',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      trainable=True)
        self.bc = self.add_weight(shape=(self.units,), initializer='zeros', name='bc', trainable=True)

        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]  # previous hidden state
        c_tm1 = states[1]  # previous cell state

        # scaling version
        i = tf.nn.relu(self.DWi * K.dot(inputs, self.Wi) + self.DWi * K.dot(h_tm1, self.Ui) + self.bi)
        f = tf.nn.relu(self.DWf * K.dot(inputs, self.Wi) + self.DWf * K.dot(h_tm1, self.Ui) + self.bf)
        o = tf.nn.relu(self.DWo * K.dot(inputs, self.Wi) + self.DWo * K.dot(h_tm1, self.Ui) + self.bo)
        c = tf.nn.relu(self.DWc * K.dot(inputs, self.Wi) + self.DWc * K.dot(h_tm1, self.Ui) + self.bc)

        new_c = f * c_tm1 + i * c
        new_h = o * tf.nn.relu(new_c)

        return new_h, [new_h, new_c]

    def get_config(self):
        config = super(MyICLSTMCell, self).get_config()
        config.update({"units": self.units})
        return config


# #进行了修改防止梯度爆炸，具体修改方法为：
# '''
# 给各个门单独定义了权重：
# 比如 self.Wi, self.Ui, self.bi, self.DWi 对应输入门；
# self.Wf, self.Uf, self.bf, self.DWf 对应遗忘门；
# self.Wo, self.Uo, self.bo, self.DWo 对应输出门；
# self.Wc, self.Uc, self.bc, self.DWc 对应候选记忆。
# call 函数中：
# 分别用 (Wi, Ui, bi, DWi) 等来计算各自门的激活，而不是都用同一组矩阵。
# 其余逻辑（ReLU、非负约束、缩放因子等）基本原封不动，只做了拆分。
# '''
# class MyICLSTMCell(tf.keras.layers.Layer):
#     def __init__(self, units, **kwargs):
#         self.units = units
#         # state_size = [h, c]
#         self.state_size = [self.units, self.units]
#         super(MyICLSTMCell, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # 输入向量维度
#         input_dim = input_shape[-1]

#         # -- Gate i (input gate) --
#         self.Wi = self.add_weight(
#             shape=(input_dim, self.units),
#             initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
#             name='Wi',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )
#         self.Ui = self.add_weight(
#             shape=(self.units, self.units),
#             initializer=initializers.Orthogonal(gain=0.1),
#             name='Ui',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )
#         self.bi = self.add_weight(
#             shape=(self.units,),
#             initializer='zeros',
#             name='bi',
#             trainable=True
#         )
#         self.DWi = self.add_weight(
#             shape=(self.units,),
#             initializer=initializers.RandomUniform(minval=0, maxval=1),
#             name='DWi',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )

#         # -- Gate f (forget gate) --
#         self.Wf = self.add_weight(
#             shape=(input_dim, self.units),
#             initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
#             name='Wf',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )
#         self.Uf = self.add_weight(
#             shape=(self.units, self.units),
#             initializer=initializers.Orthogonal(gain=0.1),
#             name='Uf',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )
#         self.bf = self.add_weight(
#             shape=(self.units,),
#             initializer='zeros',
#             name='bf',
#             trainable=True
#         )
#         self.DWf = self.add_weight(
#             shape=(self.units,),
#             initializer=initializers.RandomUniform(minval=0, maxval=1),
#             name='DWf',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )

#         # -- Gate o (output gate) --
#         self.Wo = self.add_weight(
#             shape=(input_dim, self.units),
#             initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
#             name='Wo',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )
#         self.Uo = self.add_weight(
#             shape=(self.units, self.units),
#             initializer=initializers.Orthogonal(gain=0.1),
#             name='Uo',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )
#         self.bo = self.add_weight(
#             shape=(self.units,),
#             initializer='zeros',
#             name='bo',
#             trainable=True
#         )
#         self.DWo = self.add_weight(
#             shape=(self.units,),
#             initializer=initializers.RandomUniform(minval=0, maxval=1),
#             name='DWo',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )

#         # -- Gate c (candidate) --
#         self.Wc = self.add_weight(
#             shape=(input_dim, self.units),
#             initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
#             name='Wc',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )
#         self.Uc = self.add_weight(
#             shape=(self.units, self.units),
#             initializer=initializers.Orthogonal(gain=0.1),
#             name='Uc',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )
#         self.bc = self.add_weight(
#             shape=(self.units,),
#             initializer='zeros',
#             name='bc',
#             trainable=True
#         )
#         self.DWc = self.add_weight(
#             shape=(self.units,),
#             initializer=initializers.RandomUniform(minval=0, maxval=1),
#             name='DWc',
#             constraint=tf.keras.constraints.NonNeg(),
#             trainable=True
#         )

#         self.built = True

#     def call(self, inputs, states):
#         # previous hidden state (h_{t-1})
#         h_tm1 = states[0]
#         # previous cell state (c_{t-1})
#         c_tm1 = states[1]

#         # -- 分别计算各个门(都用ReLU) --
#         i = tf.nn.relu(
#             self.DWi * (K.dot(inputs, self.Wi) + K.dot(h_tm1, self.Ui)) + self.bi
#         )
#         f = tf.nn.relu(
#             self.DWf * (K.dot(inputs, self.Wf) + K.dot(h_tm1, self.Uf)) + self.bf
#         )
#         o = tf.nn.relu(
#             self.DWo * (K.dot(inputs, self.Wo) + K.dot(h_tm1, self.Uo)) + self.bo
#         )
#         c_candidate = tf.nn.relu(
#             self.DWc * (K.dot(inputs, self.Wc) + K.dot(h_tm1, self.Uc)) + self.bc
#         )

#         # -- LSTM 状态更新 --
#         new_c = f * c_tm1 + i * c_candidate
#         new_h = o * tf.nn.relu(new_c)

#         return new_h, [new_h, new_c]

#     def get_config(self):
#         config = super(MyICLSTMCell, self).get_config()
#         config.update({"units": self.units})
#         return config