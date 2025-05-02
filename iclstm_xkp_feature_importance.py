import shap
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ICLSTM import MyICLSTMCell

# 读取数据
data = pd.read_csv('simulation_output_0325.csv')

# 配置
sequence_length = 10
predict_columns = ['Z01_T', 'Z02_T', 'Z03_T', 'Z04_T', 'Z05_T', 
                   'Z06_T', 'Z07_T', 'Z08_T', 'Bd_FracCh_Bat', 'Fa_Pw_Prod', 'Fa_E_All', 'Fa_E_HVAC']

# 这里得到“原始”的输入列（未翻倍之前）
original_input_columns = [col for col in data.columns]
# original_input_columns should be half of your eventual dimension if you're concatenating (X, -X) below.
# For example, if original_input_columns has length 114, then after doubling there will be 228.

input_data = data[original_input_columns].values
target_data = data[predict_columns].values

# 构建序列数据
X_list, Y_list = [], []
for i in range(len(data) - sequence_length):
    X_list.append(input_data[i: i + sequence_length])
    Y_list.append(target_data[i + sequence_length])

X = np.array(X_list)
Y = np.array(Y_list)

# 扩展：将 (X, -X) 连接到一起
X = np.concatenate((X, -X), axis=2)  # Now the last dimension is doubled

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.3, 
                                                    random_state=123,
                                                    shuffle=False)

# 自动推断最终特征维度，避免硬编码
num_step = sequence_length
num_target = len(predict_columns)
num_dims = X_train.shape[-1]  # This should be 2 * len(original_input_columns)

# 标准化
scaler_X = StandardScaler().fit(X_train.reshape(-1, num_dims))
X_train_scaled = scaler_X.transform(X_train.reshape(-1, num_dims)).reshape(-1, num_step, num_dims)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, num_dims)).reshape(-1, num_step, num_dims)

# 加载之前训练好的模型 (确保它的输入维度与上面对得上)
model = tf.keras.models.load_model(
    'iclstm.h5',
    compile=False,
    custom_objects={'MyICLSTMCell': MyICLSTMCell}
)

# ========== SHAP 特征重要性分析 ==========
# 对部分训练集做背景
explainer = shap.GradientExplainer(model, X_train_scaled[:500])

# 对部分测试集做分析
shap_values = explainer.shap_values(X_test_scaled[:500])
# shap_values 是 list，每个元素对应一个目标输出的 SHAP 贡献 (batch_size, 40, num_dims)

# 构造“翻倍”后的特征名，用于对齐翻倍的维度
doubled_feature_names = original_input_columns + [f'-{col}' for col in original_input_columns]
# doubled_feature_names 的长度应与 num_dims 一致

all_targets_feature_importance = []

for target_idx, target_col in enumerate(predict_columns):
    # 取当前目标的 SHAP 值
    shap_values_target = shap_values[:, :, :, target_idx]    # shape: (batch_size, 40, num_dims)

    # 求平均绝对值：得到每个输入特征对该目标的平均贡献
    feature_importance = np.mean(np.abs(shap_values_target), axis=(0, 1))  # shape: (num_dims,)

    # 构造 DataFrame
    feature_importance_df = pd.DataFrame({
        'Target': target_col,
        'Feature': doubled_feature_names,
        'Importance': feature_importance
    })

    # 排序
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    all_targets_feature_importance.append(feature_importance_df)

# 合并所有目标
final_feature_importance_df = pd.concat(all_targets_feature_importance, ignore_index=True)

# 保存结果
final_feature_importance_df.to_csv('feature_importance_from_pretrained.csv', index=False)

print("Done! SHAP-based feature importance saved to feature_importance_from_pretrained.csv.")
