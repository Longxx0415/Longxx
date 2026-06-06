import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest, f_regression

# ==================== 配置区域 ====================
DATA_PATH = 'data.csv'          # Data矩阵来源
PUREDATA_PATH = 'puredata.csv'  # Output、Input1来源

# Input1 与 Data 同结构：从"本科生"到"一年制科研助理"'在编-工勤','不可上报高研院',
INPUT1_COLS = [
    '本科生', '研究生', '党员人数', '在编-教师', '在编-教辅', '在编-思政',
    '在编-管理',  '博士后', '双轨制', '学校经费派遣',
    '部门经费派遣', '项目经费派遣', '可上报高研院',
    '可上报柔性', '不可上报柔性', '劳动合同制专职科研人员', '一年制科研助理'
]

# Output矩阵：正科、副科两列
OUTPUT_COLS = ['正科', '副科']

# Input2：学科实力、是否公共教学
INPUT2_COLS = ['学科实力指数', '是否公共教学', '学科热度']

# PCA参数
N_COMPONENTS = 6
RANDOM_STATE = 42
# ================================================


# 读取数据
df_data = pd.read_csv(DATA_PATH)
df_pure = pd.read_csv(PUREDATA_PATH)

# =================== 步骤1：划分矩阵 ===================
Data   = df_data[INPUT1_COLS].values.astype(float)      # Data[i][j]，来自data.csv
Output = df_pure[OUTPUT_COLS].values.astype(float)      # Output[i][j]，j=0正科，j=1副科
Input1 = df_pure[INPUT1_COLS].values.astype(float)      # Input1[i][j]，与Data同结构，来自puredata.csv
Input2 = df_pure[INPUT2_COLS].values.astype(float)
Data2   = df_data[INPUT2_COLS].values.astype(float)

n_depts = Data.shape[0]
print(f"数据规模：{n_depts} 个部门，Input1/Data 特征数：{len(INPUT1_COLS)}")

# =================== 步骤2：稀疏PCA求工作量矩阵 H ===================
H = np.zeros((n_depts, 2))  # H[i][j]，j=0正科，j=1副科
for j in range(2):
    print(f"\n>>> 步骤2-{j+1}：对 Input1 做PCA（对应 Output 第{j+1}列：{OUTPUT_COLS[j]}）...")

    # 标准化 Input1（去量纲）
    scaler = StandardScaler()
    Input1_scaled = scaler.fit_transform(Input1)

    # 正常PCA（无稀疏惩罚，所有变量均参与）
    pca = PCA(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )
    pca.fit(Input1_scaled)

    # 主成分载荷（shape: n_components × n_features）
    components = pca.components_

    # 方差贡献率（对应论文 βk = λk / Σλj）
    var_ratio = pca.explained_variance_ratio_

    # 统计信息
    n_features = components.shape[1]
    print(f"    保留主成分：{N_COMPONENTS}个，原始特征：{n_features}个")
    print(f"    方差贡献率：{np.round(var_ratio, 4)}")
    print(f"    累计贡献率：{np.round(var_ratio.sum(), 4)}")

    # 关键修改：H直接由Input1计算（非Data）
    # 将Input1代入主成分公式求得分，并按方差贡献率加权合成 Hi
    scores_input = Input1 @ components.T
    H[:, j] = scores_input @ var_ratio

H1 = np.zeros((n_depts, 2))  # H1[i][j]，j=0正科，j=1副科
for j in range(2):

    # 标准化 Input1（去量纲）
    scaler = StandardScaler()
    Input1_scaled = scaler.fit_transform(Input1)

    # 正常PCA（无稀疏惩罚，所有变量均参与）
    pca = PCA(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )
    pca.fit(Input1_scaled)

    # 主成分载荷（shape: n_components × n_features）
    components = pca.components_

    # 方差贡献率
    var_ratio = pca.explained_variance_ratio_

    # 关键修改：H直接由Input1计算（非Data）
    # 将Input1代入主成分公式求得分，并按方差贡献率加权合成 Hi
    scores_input = Data @ components.T
    H1[:, j] = scores_input @ var_ratio
# =================== 步骤3：编制职能强度指数 h ===================
# 先初始化 h_matrix 为 0
h_matrix = np.zeros((n_depts, 2))

# 只对 Output > 0 的位置计算 h = H / Output
for j in range(2):
    mask_positive = Output[:, j] > 0
    h_matrix[mask_positive, j] = H[mask_positive, j] / Output[mask_positive, j]

# 对 Output == 0 的位置，用该列其他非零值的平均值回填
for j in range(2):
    mask_zero = Output[:, j] == 0
    if np.any(mask_zero):
        non_zero_mean = np.mean(h_matrix[~mask_zero, j])
        h_matrix[mask_zero, j] = non_zero_mean
        zero_count = np.sum(mask_zero)
        print(f"\n>>> Output第{j+1}列（{OUTPUT_COLS[j]}）有 {zero_count} 个零值，"
              f"已用非零均值 {non_zero_mean:.6f} 回填")

# =================== 步骤4：平均职能强度指数 h_bar ===================
h1 = np.mean(h_matrix[:, 0])  # 正科平均强度
h2 = np.mean(h_matrix[:, 1])  # 副科平均强度
print(f"\n=== 步骤4：平均职能强度指数 ===")
print(f"正科平均强度 h1 = {h1:.6f}")
print(f"副科平均强度 h2 = {h2:.6f}")

# =================== 步骤5：初始编制配置矩阵 Q ===================
# Q[i][j] = H[i][j] / h_j
h_avg = np.array([h1, h2])
Q1 = H1 / h_avg
Q1_prime = H / h_avg  # 基于Data的Q1预测

# =================== 步骤6：Q2逐步回归预测 ===================
print(f"\n{'=' * 50}")
print(">>> 步骤6：基于Input2逐步回归的Q2预测")
print(f"{'=' * 50}")

Q2 = np.zeros((n_depts, 2))  # Q2[i][j]，j=0正科，j=1副科
Q2_prime = np.zeros((n_depts, 2))  # Q2[i][j]，j=0正科，j=1副科
selected_features = {}  # 记录每轮选中的特征

for j in range(2):
    y = Output[:, j]
    X = pd.DataFrame(Input2, columns=INPUT2_COLS)

    print(f"\n--- 逐步回归：Output第{j + 1}列（{OUTPUT_COLS[j]}）---")

    # 使用LassoCV实现自动特征选择（L1正则化自动将不重要系数压为0）
    # 这等价于逐步回归的"自动筛选显著变量"效果
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=2000)
    lasso.fit(X, y)

    # 获取非零系数对应的特征（被选中的变量）
    coef = lasso.coef_
    mask_selected = coef != 0
    selected_names = [INPUT2_COLS[i] for i in range(len(INPUT2_COLS)) if mask_selected[i]]

    print(f"    LassoCV最优alpha: {lasso.alpha_:.6f}")
    print(f"    选中变量: {selected_names if selected_names else '无（全部系数为0，使用全部变量）'}")
    print(f"    系数: {dict(zip(INPUT2_COLS, np.round(coef, 4)))}")

    # 如果Lasso未选中任何变量（全为0），退化为使用全部变量+最小二乘
    if not any(mask_selected):
        print(f"    [警告] Lasso未选中变量，使用全部变量进行OLS回归")
        X_selected = X
        selected_names = INPUT2_COLS.copy()
    else:
        X_selected = X.iloc[:, mask_selected]

    # 用选中的变量做OLS回归，获取精确系数和显著性
    X_ols = sm.add_constant(X_selected)
    ols_model = sm.OLS(y, X_ols).fit()
    print(f"    OLS R²: {ols_model.rsquared:.4f}")
    print(f"    回归系数（含常数项）:")
    for name, c in zip(['常数项'] + selected_names, ols_model.params):
        print(f"        {name}: {c:.6f}")

    # 记录选中特征
    selected_features[OUTPUT_COLS[j]] = selected_names

    # 用回归公式代入Data2进行预测
    # Data2需使用与Input2相同的列结构和选中变量
    Data2_df = pd.DataFrame(Data2, columns=INPUT2_COLS)
    Input2_df = pd.DataFrame(Input2, columns=INPUT2_COLS)

    if not any(mask_selected):
        Data2_selected = Data2_df
        Input2_selected = Input2_df
    else:
        Data2_selected = Data2_df.iloc[:, mask_selected]
        Input2_selected = Input2_df.iloc[:, mask_selected]

    Data2_ols = sm.add_constant(Data2_selected)
    Input2_ols = sm.add_constant(Input2_selected)
    Q2[:, j] = Data2_ols @ ols_model.params
    Q2_prime[:, j] = Input2_ols @ ols_model.params

print(f"\n{'=' * 50}")
print(">>> 逐步回归选中特征汇总")
for col, features in selected_features.items():
    print(f"    {col}: {features if features else '全部变量（Lasso未筛选出）'}")

# =================== 步骤7：方差倒数法定权 + 组合预测 ===================
print(f"\n{'='*50}")
print(">>> 步骤7：方差倒数法定权与组合预测")
print(f"{'='*50}")

# 计算Q1、Q2在训练集（puredata）上的预测误差方差
errors_Q1 = Q1_prime - Output  # Q1拟合误差
errors_Q2 = Q2_prime - Output  # Q2拟合误差

var_Q1 = np.var(errors_Q1, axis=0)  # [正科方差, 副科方差]
var_Q2 = np.var(errors_Q2, axis=0)

print(f"\nQ1拟合误差方差：正科={var_Q1[0]:.4f}, 副科={var_Q1[1]:.4f}")
print(f"Q2拟合误差方差：正科={var_Q2[0]:.4f}, 副科={var_Q2[1]:.4f}")

# 方差倒数定权
w_Q1 = (1 / var_Q1) / (1 / var_Q1 + 1 / var_Q2)
w_Q2 = (1 / var_Q2) / (1 / var_Q1 + 1 / var_Q2)

print(f"\n组合权重（方差倒数法）：")
print(f"    Q1权重：正科={w_Q1[0]:.4f}, 副科={w_Q1[1]:.4f}")
print(f"    Q2权重：正科={w_Q2[0]:.4f}, 副科={w_Q2[1]:.4f}")

# 组合预测：Q_combined = w1*Q1' + w2*Q2'
Q_combined = np.zeros((n_depts, 2))
Q_combined[:, 0] = w_Q1[0] * Q1[:, 0] + w_Q2[0] * Q2[:, 0]  # 正科
Q_combined[:, 1] = w_Q1[1] * Q1[:, 1] + w_Q2[1] * Q2[:, 1]  # 副科

# =================== 结果输出 ===================
result = pd.DataFrame({
    '部门名称': df_pure['部门名称'].values,

    # 实际编制
    '实有正科': Output[:, 0].astype(int),
    '实有副科': Output[:, 1].astype(int),

    # Q1：基于Input1的PCA拟合值（训练集）
    'Q1_正科': np.round(Q1[:, 0], 2),
    'Q1_副科': np.round(Q1[:, 1], 2),
    'Q1正科差额': np.round(Q1[:, 0] - Output[:, 0], 2),
    'Q1副科差额': np.round(Q1[:, 1] - Output[:, 1], 2),

    # Q2：基于Data2的回归预测值（新样本）
    'Q2_正科': np.round(Q2[:, 0], 2),
    'Q2_副科': np.round(Q2[:, 1], 2),
    'Q2正科差额': np.round(Q2[:, 0] - Output[:, 0], 2),
    'Q2副科差额': np.round(Q2[:, 1] - Output[:, 1], 2),

    # Q_combined：组合预测（最终输出）
    'Qc_正科': np.round(Q_combined[:, 0], 2),
    'Qc_副科': np.round(Q_combined[:, 1], 2),
    'Qc正科差额': np.round(Q_combined[:, 0] - Output[:, 0], 2),
    'Qc副科差额': np.round(Q_combined[:, 1] - Output[:, 1], 2),


})

# 添加合计行
summary = pd.DataFrame([{
    '部门名称': '合计/平均',

    '实有正科': int(Output[:, 0].sum()),
    '实有副科': int(Output[:, 1].sum()),

    'Q1_正科': np.round(Q1[:, 0].sum(), 2),
    'Q1_副科': np.round(Q1[:, 1].sum(), 2),
    'Q1正科差额': np.round(Q1[:, 0].sum() - Output[:, 0].sum(), 2),
    'Q1副科差额': np.round(Q1[:, 1].sum() - Output[:, 1].sum(), 2),

    'Q2_正科': np.round(Q2[:, 0].sum(), 2),
    'Q2_副科': np.round(Q2[:, 1].sum(), 2),
    'Q2正科差额': np.round(Q2[:, 0].sum() - Output[:, 0].sum(), 2),
    'Q2副科差额': np.round(Q2[:, 1].sum() - Output[:, 1].sum(), 2),

    'Qc_正科': np.round(Q_combined[:, 0].sum(), 2),
    'Qc_副科': np.round(Q_combined[:, 1].sum(), 2),
    'Qc正科差额': np.round(Q_combined[:, 0].sum() - Output[:, 0].sum(), 2),
    'Qc副科差额': np.round(Q_combined[:, 1].sum() - Output[:, 1].sum(), 2),


}])
result = pd.concat([result, summary], ignore_index=True)

print("\n=== 结果预览 ===")
print(result.to_string(index=False))

# 保存
result.to_csv('result_编制预测.csv', index=False, encoding='utf-8-sig')
print(f"\n[完成] 结果已保存至：result_编制预测.csv")