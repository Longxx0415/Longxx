import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
from itertools import combinations
import os

# ==================== 配置区域 ====================
TRAIN_PATH = 'puredata.csv'
TEST_PATH = 'data.csv'
FORECAST_PATH = 'forecastdata.xlsx'

Q1_COLS = ['本科生', '研究生', '党员人数', '在编-教师']

# Q2候选池：4基础 + 3交互 + 3非线性（平方项）
Q2_BASE = ['学科实力指数', '是否公共教学', '学科热度', '教师变动规模']
Q2_INTERACT = [
    '是否公共教学_学科实力',
    '是否公共教学_学科热度',
    '是否公共教学_教师变动',
]
Q2_POLY = [
    '学科实力指数_sq',
    '学科热度_sq',
    '教师变动规模_sq',
]
Q2_ALL_COLS = Q2_BASE + Q2_INTERACT + Q2_POLY

# 拆分比例：正科:副科:科级以下 = 2:3:5
RATIOS = np.array([0.2, 0.3, 0.5])
RATIO_NAMES = ['正科(20%)', '副科(30%)', '科级以下(50%)']
OUTPUT_COLS = ['正科', '副科', '科级以下']

N_COMPONENTS = 3
RANDOM_STATE = 42
ALPHA = 0.05
SUBSET_CRITERION = 'bic'
# ================================================


def build_q2_features(df):
    """构建Q2候选特征：基础 + 交互 + 非线性平方项"""
    df = df.copy()
    df['是否公共教学_学科实力'] = df['是否公共教学'] * df['学科实力指数']
    df['是否公共教学_学科热度'] = df['是否公共教学'] * df['学科热度']
    df['是否公共教学_教师变动'] = df['是否公共教学'] * df['教师变动规模']
    df['学科实力指数_sq'] = df['学科实力指数'] ** 2
    df['学科热度_sq'] = df['学科热度'] ** 2
    df['教师变动规模_sq'] = df['教师变动规模'] ** 2
    return df


def budget_adjust(total_pred, B, ratios=RATIOS):
    """
    预算约束优化：对总编制进行约束，再按比例拆分
    情形1: total <= B, 不变
    情形2: total > B, 总编制压缩到B，再按比例拆分
    """
    total_pred = float(total_pred)
    if total_pred <= B + 1e-9:
        return total_pred * ratios
    # 超预算，总编制压缩到预算
    return B * ratios


def best_subset_selection(X, y, criterion='bic'):
    """
    最优子集选择：遍历所有特征组合，强制要求包含平方项时必须同时包含对应基础项
    """
    n_features = X.shape[1]
    feature_names = X.columns.tolist()

    # 建立平方项与基础项的映射
    poly_base_map = {}
    for f in feature_names:
        if '_sq' in f:
            base = f.replace('_sq', '')
            if base in feature_names:
                poly_base_map[f] = base

    # 生成有效组合：如果包含_x_sq，则必须同时包含_x
    def is_valid_combo(selected):
        for var in selected:
            if var in poly_base_map and poly_base_map[var] not in selected:
                return False
        return True

    if criterion in ['aic', 'bic']:
        best_score = float('inf')
    else:
        best_score = -float('inf')

    best_model = None
    best_features = []
    results = []

    for k in range(1, n_features + 1):
        for combo in combinations(range(n_features), k):
            selected_names = [feature_names[i] for i in combo]

            # 跳过无效组合（含平方项但不含对应基础项）
            if not is_valid_combo(selected_names):
                continue

            X_subset = sm.add_constant(X[selected_names])
            try:
                model = sm.OLS(y, X_subset).fit()

                if criterion == 'aic':
                    score = model.aic
                elif criterion == 'bic':
                    score = model.bic
                else:
                    score = model.rsquared_adj

                results.append({
                    'features': ', '.join(selected_names),
                    'k': k,
                    'aic': round(model.aic, 2),
                    'bic': round(model.bic, 2),
                    'adj_r2': round(model.rsquared_adj, 4),
                    'r2': round(model.rsquared, 4)
                })

                if criterion in ['aic', 'bic']:
                    improved = score < best_score
                else:
                    improved = score > best_score

                if improved:
                    best_score = score
                    best_model = model
                    best_features = selected_names
            except Exception:
                continue

    return best_model, best_features, best_score, pd.DataFrame(results)


def split_total(total, ratios=RATIOS):
    """按给定比例拆分总编制为三个层级"""
    return total * ratios


# =================== 读取数据 ===================
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

if os.path.exists('forecastdata.csv'):
    df_forecast = pd.read_csv('forecastdata.csv')
else:
    df_forecast = pd.read_excel(FORECAST_PATH)

df_train = build_q2_features(df_train)
df_test = build_q2_features(df_test)
df_forecast = build_q2_features(df_forecast)

# 计算总编制人数
df_train['总编制'] = df_train['正科'] + df_train['副科'] + df_train['科级以下']
df_test['总编制'] = df_test['正科'] + df_test['副科'] + df_test['科级以下']

n_train = len(df_train)
n_test = len(df_test)
n_forecast = len(df_forecast)

cat_train = df_train['大类'].values
cat_test = df_test['大类'].values
cat_forecast = df_forecast['大类'].values
categories = np.unique(cat_train)

Input1_train = df_train[Q1_COLS].values.astype(float)
Input1_test = df_test[Q1_COLS].values.astype(float)
Input1_forecast = df_forecast[Q1_COLS].values.astype(float)

# 总编制（新的预测目标）
Total_train = df_train['总编制'].values.astype(float)
Total_test = df_test['总编制'].values.astype(float)

# Q2标准化
scaler_q2 = StandardScaler()
Input2_train_raw = df_train[Q2_ALL_COLS].values.astype(float)
Input2_test_raw = df_test[Q2_ALL_COLS].values.astype(float)
Input2_forecast_raw = df_forecast[Q2_ALL_COLS].values.astype(float)
Input2_train = scaler_q2.fit_transform(Input2_train_raw)
Input2_test = scaler_q2.transform(Input2_test_raw)
Input2_forecast = scaler_q2.transform(Input2_forecast_raw)

Budget_test = df_test['预算约束人数'].values.astype(float)
Budget_forecast = df_forecast['预算约束人数'].values.astype(float)

print(f"{'='*60}")
print(">>> 数据读取与配置")
print(f"{'='*60}")
print(f"数据规模：训练集{n_train}，测试集{n_test}，预测集{n_forecast}")
print(f"Q1特征（PCA）：{Q1_COLS}")
print(f"Q2候选变量：{Q2_ALL_COLS}")
print(f"预测目标：总编制人数（正科+副科+科级以下）")
print(f"拆分比例：正科20% : 副科30% : 科级以下50%")
print(f"大类：{list(categories)}")

# =================== 步骤1：PCA求工作量矩阵H ===================
print(f"\n{'='*60}")
print(">>> 步骤1：PCA求工作量矩阵H")
print(f"{'='*60}")

scaler_q1 = StandardScaler()
Input1_train_scaled = scaler_q1.fit_transform(Input1_train)

pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
pca.fit(Input1_train_scaled)

components = pca.components_
var_ratio = pca.explained_variance_ratio_

print(f"\n--- 1.1 PCA拟合结果 ---")
print(f"    保留主成分：{N_COMPONENTS}个，原始特征：{len(Q1_COLS)}个")
print(f"    方差贡献率：{np.round(var_ratio, 4)}")
print(f"    累计贡献率：{np.round(var_ratio.sum(), 4)}")

loadings_df = pd.DataFrame(
    components.T,
    index=Q1_COLS,
    columns=[f'PC{k+1}' for k in range(N_COMPONENTS)]
)
print(f"\n--- 1.2 主成分载荷矩阵 ---")
print(loadings_df.round(4).to_string())

def compute_H(Input1_raw):
    scores = Input1_raw @ components.T
    H = scores @ var_ratio
    return H

H_train = compute_H(Input1_train)
H_test = compute_H(Input1_test)
H_forecast = compute_H(Input1_forecast)

print(f"\n--- 1.3 综合工作量H ---")
print(f"    H_train范围: [{H_train.min():.2f}, {H_train.max():.2f}]")

# =================== 步骤2：按大类求编制职能强度指数h ===================
print(f"\n{'='*60}")
print(">>> 步骤2：按大类求总编制职能强度指数h")
print(f"{'='*60}")

h_train = np.zeros(n_train)
mask_positive = Total_train > 0
h_train[mask_positive] = H_train[mask_positive] / Total_train[mask_positive]

# 零值按大类回填
for cat in categories:
    mask_cat = cat_train == cat
    non_zero = h_train[mask_cat][h_train[mask_cat] > 0]
    fill_val = np.mean(non_zero) if len(non_zero) > 0 else np.mean(h_train[h_train > 0])
    mask_zero = (h_train == 0) & mask_cat
    if np.sum(mask_zero) > 0:
        print(f"    {cat}类: {np.sum(mask_zero)}个零值，用非零均值{fill_val:.4f}回填")
    h_train[mask_zero] = fill_val

h_bar = {}
for cat in categories:
    mask_cat = cat_train == cat
    h_bar[cat] = np.mean(h_train[mask_cat])

print(f"\n--- 2.1 按大类平均职能强度指数 h_bar ---")
for cat in categories:
    print(f"    {cat}类: h_bar={h_bar[cat]:.4f}")

# =================== 步骤3：由职能强度预测总编制 Q1 ===================
print(f"\n{'='*60}")
print(">>> 步骤3：由职能强度预测总编制 Q1")
print(f"{'='*60}")

Q1_total_train = np.zeros(n_train)
Q1_total_test = np.zeros(n_test)
Q1_total_forecast = np.zeros(n_forecast)

for i in range(n_train):
    cat = cat_train[i]
    Q1_total_train[i] = H_train[i] / h_bar[cat] if h_bar[cat] != 0 else 0
for i in range(n_test):
    cat = cat_test[i]
    Q1_total_test[i] = H_test[i] / h_bar[cat] if h_bar[cat] != 0 else 0
for i in range(n_forecast):
    cat = cat_forecast[i]
    Q1_total_forecast[i] = H_forecast[i] / h_bar[cat] if h_bar[cat] != 0 else 0

print(f"--- 3.1 Q1预测总编制结果 ---")
print(f"    训练集: 均值={Q1_total_train.mean():.2f}, 范围=[{Q1_total_train.min():.2f}, {Q1_total_train.max():.2f}]")
print(f"    测试集: 均值={Q1_total_test.mean():.2f}, 范围=[{Q1_total_test.min():.2f}, {Q1_total_test.max():.2f}]")

# =================== 步骤4：最优子集选择 Q2（预测总编制）====================
print(f"\n{'='*60}")
print(f">>> 步骤4：最优子集选择 Q2（预测总编制）")
print(f"{'='*60}")

X_train_df = pd.DataFrame(Input2_train, columns=Q2_ALL_COLS)
ols_model, selected_names, bic, all_results = best_subset_selection(
    X_train_df, Total_train, criterion=SUBSET_CRITERION
)

r2_q2 = ols_model.rsquared

print(f"\n--- 4.1 最优模型 ---")
print(f"    选中变量: {selected_names if selected_names else '无（仅常数项）'}")
print(f"    BIC={bic:.2f}, OLS R²={r2_q2:.4f}")
print(f"    遍历模型数: {len(all_results)} / {2**len(Q2_ALL_COLS)-1}")

# 判断变量类型
poly_selected = [f for f in selected_names if '_sq' in f]
interact_selected = [f for f in selected_names if '_' in f and '_sq' not in f]
base_selected = [f for f in selected_names if f in Q2_BASE]
print(f"    基础变量: {base_selected}")
print(f"    交互项: {interact_selected if interact_selected else '无'}")
print(f"    非线性项: {poly_selected if poly_selected else '无'}")

# top3备选
top3 = all_results.nsmallest(3, 'bic')
print(f"\n    top3备选模型:")
for _, row in top3.iterrows():
    print(f"      {row['features']} → BIC={row['bic']}, R²={row['r2']}")

print(f"\n    回归系数（含常数项）:")
for name, c in ols_model.params.items():
    print(f"        {name}: {c:.6f}")

# Q2预测总编制
X_test_df = pd.DataFrame(Input2_test, columns=Q2_ALL_COLS)
X_forecast_df = pd.DataFrame(Input2_forecast, columns=Q2_ALL_COLS)

Q2_total_train = ols_model.predict(sm.add_constant(X_train_df[selected_names]))
Q2_total_test = ols_model.predict(sm.add_constant(X_test_df[selected_names]))
Q2_total_forecast = ols_model.predict(sm.add_constant(X_forecast_df[selected_names]))

print(f"\n--- 4.2 Q2预测总编制结果 ---")
print(f"    训练集: 均值={Q2_total_train.mean():.2f}, 范围=[{Q2_total_train.min():.2f}, {Q2_total_train.max():.2f}]")

# =================== 步骤5：方差倒数法定权 + 组合预测总编制 ===================
print(f"\n{'='*60}")
print(">>> 步骤5：方差倒数法定权与组合预测总编制")
print(f"{'='*60}")

errors_Q1 = Q1_total_train - Total_train
errors_Q2 = Q2_total_train - Total_train
var_Q1 = np.var(errors_Q1)
var_Q2 = np.var(errors_Q2)

w_Q1 = (1 / var_Q1) / (1 / var_Q1 + 1 / var_Q2)
w_Q2 = (1 / var_Q2) / (1 / var_Q1 + 1 / var_Q2)

print(f"--- 5.1 拟合误差方差 ---")
print(f"    Q1误差方差: {var_Q1:.4f}")
print(f"    Q2误差方差: {var_Q2:.4f}")
print(f"\n--- 5.2 组合权重 ---")
print(f"    Q1权重={w_Q1:.4f}, Q2权重={w_Q2:.4f}")

Qc_total_train = w_Q1 * Q1_total_train + w_Q2 * Q2_total_train
Qc_total_test = w_Q1 * Q1_total_test + w_Q2 * Q2_total_test
Qc_total_forecast = w_Q1 * Q1_total_forecast + w_Q2 * Q2_total_forecast

print(f"\n--- 5.3 组合预测总编制 ---")
print(f"    训练集: 均值={Qc_total_train.mean():.2f}")
print(f"    测试集: 均值={Qc_total_test.mean():.2f}")
print(f"    预测集: 均值={Qc_total_forecast.mean():.2f}")

# =================== 步骤6：按比例拆分 + 预算约束 ===================
print(f"\n{'='*60}")
print(">>> 步骤6：按比例拆分与预算约束")
print(f"{'='*60}")

# 测试集拆分（无约束）
Qc_test_split = np.zeros((n_test, 3))
for i in range(n_test):
    Qc_test_split[i] = split_total(Qc_total_test[i])

print(f"--- 6.1 测试集拆分（无约束，2:3:5）---")
for j, name in enumerate(OUTPUT_COLS):
    print(f"    {name}: 合计={Qc_test_split[:,j].sum():.2f}, 均值={Qc_test_split[:,j].mean():.2f}")

# 预测集拆分（约束前）
Qc_forecast_split = np.zeros((n_forecast, 3))
for i in range(n_forecast):
    Qc_forecast_split[i] = split_total(Qc_total_forecast[i])

print(f"\n--- 6.2 预测集拆分（约束前）---")
for j, name in enumerate(OUTPUT_COLS):
    print(f"    {name}: 合计={Qc_forecast_split[:,j].sum():.2f}")
print(f"    总编制合计: {Qc_forecast_split.sum():.2f}")

# 预测集拆分（约束后）
Qc_forecast_adj_total = np.zeros(n_forecast)
Qc_forecast_adj_split = np.zeros((n_forecast, 3))

for i in range(n_forecast):
    total_before = Qc_total_forecast[i]
    budget = Budget_forecast[i]
    Qc_forecast_adj_split[i] = budget_adjust(total_before, budget)
    Qc_forecast_adj_total[i] = Qc_forecast_adj_split[i].sum()

print(f"\n--- 6.3 预测集拆分（约束后）---")
for j, name in enumerate(OUTPUT_COLS):
    print(f"    {name}: 合计={Qc_forecast_adj_split[:,j].sum():.2f}")
print(f"    总编制合计: {Qc_forecast_adj_split.sum():.2f}")
print(f"    预算约束合计: {Budget_forecast.sum():.0f}")

# 超预算部门
print(f"\n--- 6.4 超预算部门 ---")
for i in range(n_forecast):
    total_before = Qc_total_forecast[i]
    if total_before > Budget_forecast[i] + 0.01:
        print(f"    {df_forecast.iloc[i]['部门名称']}:")
        print(f"        预算={Budget_forecast[i]:.0f}, 约束前={total_before:.2f}, 约束后={Qc_forecast_adj_total[i]:.2f}")
        print(f"        拆分: 正科={Qc_forecast_adj_split[i,0]:.2f}, 副科={Qc_forecast_adj_split[i,1]:.2f}, 科以下={Qc_forecast_adj_split[i,2]:.2f}")

# =================== 步骤7：假设检验与精度指标（总编制）====================
print(f"\n{'='*60}")
print(">>> 步骤7：测试集假设检验与预测精度指标（总编制人数）")
print(f"{'='*60}")

actual_total = Total_test
predicted_total = Qc_total_test
residual_total = predicted_total - actual_total
n = len(actual_total)

print(f"\n--- 总编制人数 ---")
print(f"    实际值: 均值={actual_total.mean():.2f}, 方差={np.var(actual_total, ddof=1):.4f}")
print(f"    预测值: 均值={predicted_total.mean():.2f}, 方差={np.var(predicted_total, ddof=1):.4f}")

# 配对t检验
t_stat, p_mean = stats.ttest_rel(predicted_total, actual_total)
mean_diff = np.mean(residual_total)
mean_conclusion = "通过" if p_mean > ALPHA else "未通过"
print(f"\n    [总体均值] 配对t检验:")
print(f"        均值差={mean_diff:.4f}, p={p_mean:.4f} → {mean_conclusion}")

# F检验
var_pred = np.var(predicted_total, ddof=1)
var_actual = np.var(actual_total, ddof=1)
if var_pred > var_actual:
    f_stat = var_pred / var_actual
    p_tail = 1 - stats.f.cdf(f_stat, n-1, n-1)
else:
    f_stat = var_actual / var_pred
    p_tail = 1 - stats.f.cdf(f_stat, n-1, n-1)
p_var_f = 2 * min(p_tail, 1-p_tail)
p_var_f = min(p_var_f, 1.0)
f_conclusion = "通过" if p_var_f > ALPHA else "未通过"
print(f"    [总体方差] F检验:")
print(f"        预测方差={var_pred:.4f}, 实际方差={var_actual:.4f}")
print(f"        F={f_stat:.4f}, p={p_var_f:.4f} → {f_conclusion}")

# Levene检验
stat_lev, p_lev = stats.levene(predicted_total, actual_total)
lev_conclusion = "通过" if p_lev > ALPHA else "未通过"
print(f"    [总体方差] Levene检验:")
print(f"        p={p_lev:.4f} → {lev_conclusion}")

# 精度指标
mae = np.mean(np.abs(residual_total))
rmse = np.sqrt(np.mean(residual_total**2))
ss_res = np.sum(residual_total**2)
ss_tot = np.sum((actual_total - np.mean(actual_total))**2)
r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
print(f"    [预测精度] MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# 保存指标
test_metrics_total = {
    'p_mean': round(p_mean, 4), 'mean_conclusion': mean_conclusion,
    'p_var_f': round(p_var_f, 4), 'f_conclusion': f_conclusion,
    'p_lev': round(p_lev, 4), 'lev_conclusion': lev_conclusion,
    'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4),
}

# 各层级精度（仅供参考）
print(f"\n--- 各层级精度（按2:3:5拆分后，仅供参考）---")
actual_split = df_test[['正科', '副科', '科级以下']].values.astype(float)
for j, col_name in enumerate(OUTPUT_COLS):
    residual_j = Qc_test_split[:, j] - actual_split[:, j]
    mae_j = np.mean(np.abs(residual_j))
    rmse_j = np.sqrt(np.mean(residual_j**2))
    print(f"    {col_name}: MAE={mae_j:.4f}, RMSE={rmse_j:.4f}")

# =================== 结果输出 ===================
print(f"\n{'='*60}")
print(">>> 最终结果输出")
print(f"{'='*60}")

# 总编制假设检验汇总表
test_summary = pd.DataFrame({
    '检验/指标': ['配对t检验(p值)', 'F检验(p值)', 'Levene检验(p值)', 'MAE', 'RMSE', 'R²'],
    '总编制人数': [
        f"{test_metrics_total['p_mean']} ({test_metrics_total['mean_conclusion']})",
        f"{test_metrics_total['p_var_f']} ({test_metrics_total['f_conclusion']})",
        f"{test_metrics_total['p_lev']} ({test_metrics_total['lev_conclusion']})",
        test_metrics_total['mae'],
        test_metrics_total['rmse'],
        test_metrics_total['r2'],
    ]
})

print("\n=== 测试集假设检验与精度指标汇总（总编制）===")
print(test_summary.to_string(index=False))

# 预测集结果（约束前后对比）
forecast_result = pd.DataFrame({
    '大类': df_forecast['大类'].values,
    '部门名称': df_forecast['部门名称'].values,
    '预算约束人数': Budget_forecast.astype(int),
    '约束前_总编制': np.round(Qc_total_forecast, 2),
    '约束前_正科': np.round(Qc_forecast_split[:, 0], 2),
    '约束前_副科': np.round(Qc_forecast_split[:, 1], 2),
    '约束前_科级以下': np.round(Qc_forecast_split[:, 2], 2),
    '约束后_总编制': np.round(Qc_forecast_adj_total, 2),
    '约束后_正科': np.round(Qc_forecast_adj_split[:, 0], 2),
    '约束后_副科': np.round(Qc_forecast_adj_split[:, 1], 2),
    '约束后_科级以下': np.round(Qc_forecast_adj_split[:, 2], 2),
})

forecast_summary = pd.DataFrame([{
    '大类': '合计',
    '部门名称': '-',
    '预算约束人数': int(Budget_forecast.sum()),
    '约束前_总编制': np.round(Qc_total_forecast.sum(), 2),
    '约束前_正科': np.round(Qc_forecast_split[:, 0].sum(), 2),
    '约束前_副科': np.round(Qc_forecast_split[:, 1].sum(), 2),
    '约束前_科级以下': np.round(Qc_forecast_split[:, 2].sum(), 2),
    '约束后_总编制': np.round(Qc_forecast_adj_total.sum(), 2),
    '约束后_正科': np.round(Qc_forecast_adj_split[:, 0].sum(), 2),
    '约束后_副科': np.round(Qc_forecast_adj_split[:, 1].sum(), 2),
    '约束后_科级以下': np.round(Qc_forecast_adj_split[:, 2].sum(), 2),
}])
forecast_result = pd.concat([forecast_result, forecast_summary], ignore_index=True)

print("\n=== 预测集约束前与约束后结果 ===")
print(forecast_result.to_string(index=False))

# 保存
script_dir = os.path.dirname(os.path.abspath(__file__))

def safe_save(df, filename):
    save_path = os.path.join(script_dir, filename)
    counter = 1
    while os.path.exists(save_path):
        try:
            with open(save_path, 'a'):
                pass
            break
        except PermissionError:
            save_path = os.path.join(script_dir, f'{filename[:-4]}_{counter}.csv')
            counter += 1
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    return save_path

save_path_test = safe_save(test_summary, 'result_total_test_metrics.csv')
save_path_forecast = safe_save(forecast_result, 'result_total_forecast.csv')

print(f"\n{'='*60}")
print("[完成] 结果保存")
print(f"{'='*60}")
print(f"测试集指标已保存至：{save_path_test}")
print(f"预测集结果已保存至：{save_path_forecast}")
