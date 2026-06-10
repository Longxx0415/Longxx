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


def budget_adjust(y_pred, B):
    """预算约束优化"""
    y_pred = np.array(y_pred, dtype=float)
    total = y_pred.sum()
    if total <= B + 1e-9:
        return y_pred
    excess = total - B
    adjusted = y_pred - excess / 3.0
    for _ in range(10):
        neg_mask = adjusted < 0
        if not np.any(neg_mask):
            break
        freed = np.sum(np.abs(adjusted[neg_mask]))
        adjusted[neg_mask] = 0
        pos_mask = adjusted > 0
        if np.any(pos_mask):
            adjusted[pos_mask] -= freed / np.sum(pos_mask)
    return np.maximum(adjusted, 0)


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
        compare = lambda s, best: s < best
    else:
        best_score = -float('inf')
        compare = lambda s, best: s > best

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

                if compare(score, best_score):
                    best_score = score
                    best_model = model
                    best_features = selected_names
            except Exception:
                continue

    return best_model, best_features, best_score, pd.DataFrame(results)


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

Output_train = df_train[OUTPUT_COLS].values.astype(float)
Output_test = df_test[OUTPUT_COLS].values.astype(float)

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

print(f"{'=' * 60}")
print(">>> 数据读取与配置")
print(f"{'=' * 60}")
print(f"数据规模：训练集{n_train}，测试集{n_test}，预测集{n_forecast}")
print(f"Q1特征（PCA）：{Q1_COLS}")
print(f"Q2候选变量（含非线性项）：{Q2_ALL_COLS}")
print(f"编制类别：{OUTPUT_COLS}")
print(f"大类：{list(categories)}")

# =================== 步骤1：PCA求工作量矩阵H ===================
print(f"\n{'=' * 60}")
print(">>> 步骤1：PCA求工作量矩阵H")
print(f"{'=' * 60}")

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
    columns=[f'PC{k + 1}' for k in range(N_COMPONENTS)]
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

H_train = np.tile(H_train.reshape(-1, 1), (1, 3))
H_test = np.tile(H_test.reshape(-1, 1), (1, 3))
H_forecast = np.tile(H_forecast.reshape(-1, 1), (1, 3))

# =================== 步骤2：按大类求编制职能强度指数h ===================
print(f"\n{'=' * 60}")
print(">>> 步骤2：按大类求编制职能强度指数h")
print(f"{'=' * 60}")

h_matrix_train = np.zeros((n_train, 3))

for j in range(3):
    mask_positive = Output_train[:, j] > 0
    h_matrix_train[mask_positive, j] = H_train[mask_positive, j] / Output_train[mask_positive, j]

for j in range(3):
    for cat in categories:
        mask_cat = cat_train == cat
        non_zero_vals = h_matrix_train[mask_cat, j][h_matrix_train[mask_cat, j] > 0]
        if len(non_zero_vals) > 0:
            fill_val = np.mean(non_zero_vals)
        else:
            global_non_zero = h_matrix_train[:, j][h_matrix_train[:, j] > 0]
            fill_val = np.mean(global_non_zero) if len(global_non_zero) > 0 else 0
        mask_zero = (h_matrix_train[:, j] == 0) & mask_cat
        h_matrix_train[mask_zero, j] = fill_val

h_bar = {}
for cat in categories:
    h_bar[cat] = {}
    mask_cat = cat_train == cat
    for j in range(3):
        vals = h_matrix_train[mask_cat, j]
        h_bar[cat][j] = np.mean(vals)

print(f"--- 2.1 按大类平均职能强度指数 h_bar ---")
for cat in categories:
    print(f"    {cat}类：正科={h_bar[cat][0]:.4f}, 副科={h_bar[cat][1]:.4f}, 科级以下={h_bar[cat][2]:.4f}")

# =================== 步骤3：由职能强度预测编制 Q1 ===================
print(f"\n{'=' * 60}")
print(">>> 步骤3：由职能强度预测编制 Q1")
print(f"{'=' * 60}")


def predict_Q1(H_matrix, cat_vector):
    Q1 = np.zeros_like(H_matrix)
    for i in range(len(H_matrix)):
        cat = cat_vector[i]
        for j in range(3):
            Q1[i, j] = H_matrix[i, j] / h_bar[cat][j] if h_bar[cat][j] != 0 else 0
    return Q1


Q1_train = predict_Q1(H_train, cat_train)
Q1_test = predict_Q1(H_test, cat_test)
Q1_forecast = predict_Q1(H_forecast, cat_forecast)

# =================== 步骤4：标准最优子集选择 Q2 ===================
print(f"\n{'=' * 60}")
print(f">>> 步骤4：标准最优子集选择（OLS直接拟合，{SUBSET_CRITERION.upper()}定权）")
print(f"{'=' * 60}")

Q2_train = np.zeros((n_train, 3))
Q2_test = np.zeros((n_test, 3))
Q2_forecast = np.zeros((n_forecast, 3))
selected_features = {}

for j in range(3):
    y = Output_train[:, j]
    X_train_df = pd.DataFrame(Input2_train, columns=Q2_ALL_COLS)

    print(f"\n--- 4.{j + 1} 最优子集选择：{OUTPUT_COLS[j]} ---")

    ols_model, selected_names, best_score, all_results = best_subset_selection(
        X_train_df, y, criterion=SUBSET_CRITERION
    )

    r2_q2 = ols_model.rsquared

    print(f"    最优模型变量: {selected_names if selected_names else '无（仅常数项）'}")
    print(f"    {SUBSET_CRITERION.upper()}={best_score:.2f}, OLS R²={r2_q2:.4f}")
    print(f"    遍历模型数: {len(all_results)} / {2 ** len(Q2_ALL_COLS) - 1}")

    # 判断变量类型
    poly_selected = [f for f in selected_names if '_sq' in f]
    interact_selected = [f for f in selected_names if '_' in f and '_sq' not in f]
    base_selected = [f for f in selected_names if f in Q2_BASE]
    print(f"    基础变量: {base_selected}")
    print(f"    交互项: {interact_selected if interact_selected else '无'}")
    print(f"    非线性项: {poly_selected if poly_selected else '无'}")

    # 打印top3备选
    if SUBSET_CRITERION == 'bic':
        top3 = all_results.nsmallest(3, 'bic')
    elif SUBSET_CRITERION == 'aic':
        top3 = all_results.nsmallest(3, 'aic')
    else:
        top3 = all_results.nlargest(3, 'adj_r2')
    print(f"    top3备选模型:")
    for _, row in top3.iterrows():
        print(f"      {row['features']} → BIC={row['bic']}, R²={row['r2']}")

    print(f"    回归系数（含常数项）:")
    for name, c in ols_model.params.items():
        print(f"        {name}: {c:.6f}")

    selected_features[OUTPUT_COLS[j]] = selected_names

    # 预测
    X_test_df = pd.DataFrame(Input2_test, columns=Q2_ALL_COLS)
    X_forecast_df = pd.DataFrame(Input2_forecast, columns=Q2_ALL_COLS)

    if selected_names:
        X_test_const = sm.add_constant(X_test_df[selected_names])
        X_forecast_const = sm.add_constant(X_forecast_df[selected_names])
    else:
        X_test_const = sm.add_constant(pd.DataFrame(index=range(n_test)))
        X_forecast_const = sm.add_constant(pd.DataFrame(index=range(n_forecast)))

    Q2_train[:, j] = ols_model.predict(ols_model.model.exog)
    Q2_test[:, j] = ols_model.predict(X_test_const)
    Q2_forecast[:, j] = ols_model.predict(X_forecast_const)

print(f"\n{'=' * 60}")
print(">>> 步骤4汇总：Q2最优模型")
print(f"{'=' * 60}")
for col, features in selected_features.items():
    print(f"    {col}: {features if features else '仅常数项'}")

# =================== 步骤5：方差倒数法定权 + 组合预测 ===================
print(f"\n{'=' * 60}")
print(">>> 步骤5：方差倒数法定权与组合预测")
print(f"{'=' * 60}")

errors_Q1_train = Q1_train - Output_train
errors_Q2_train = Q2_train - Output_train

var_Q1 = np.var(errors_Q1_train, axis=0)
var_Q2 = np.var(errors_Q2_train, axis=0)

w_Q1 = (1 / var_Q1) / (1 / var_Q1 + 1 / var_Q2)
w_Q2 = (1 / var_Q2) / (1 / var_Q1 + 1 / var_Q2)

print(f"--- 5.1 拟合误差方差 ---")
print(f"    Q1误差方差：正科={var_Q1[0]:.4f}, 副科={var_Q1[1]:.4f}, 科级以下={var_Q1[2]:.4f}")
print(f"    Q2误差方差：正科={var_Q2[0]:.4f}, 副科={var_Q2[1]:.4f}, 科级以下={var_Q2[2]:.4f}")

print(f"\n--- 5.2 组合权重（方差倒数法）---")
for j, name in enumerate(OUTPUT_COLS):
    print(f"    {name}: Q1权重={w_Q1[j]:.4f}, Q2权重={w_Q2[j]:.4f}")

Qc_train = w_Q1 * Q1_train + w_Q2 * Q2_train
Qc_test = w_Q1 * Q1_test + w_Q2 * Q2_test
Qc_forecast = w_Q1 * Q1_forecast + w_Q2 * Q2_forecast

print(f"\n--- 5.3 组合预测结果 ---")
for j, col_name in enumerate(OUTPUT_COLS):
    print(f"    {col_name}: 训练集均值={Qc_train[:, j].mean():.2f}, 测试集均值={Qc_test[:, j].mean():.2f}")

# =================== 步骤6：预算约束优化（仅预测集） ===================
print(f"\n{'=' * 60}")
print(">>> 步骤6：预算约束优化调整（仅预测集）")
print(f"{'=' * 60}")

print(f"--- 6.1 测试集预测合计（无约束）---")
print(f"    正科={Qc_test[:, 0].sum():.2f}, 副科={Qc_test[:, 1].sum():.2f}, "
      f"科级以下={Qc_test[:, 2].sum():.2f}, 总计={Qc_test.sum():.2f}")

Qc_forecast_adj = np.zeros_like(Qc_forecast)
for i in range(n_forecast):
    Qc_forecast_adj[i] = budget_adjust(Qc_forecast[i], Budget_forecast[i])

print(f"\n--- 6.2 预测集约束前后对比 ---")
print(f"    约束前：正科={Qc_forecast[:, 0].sum():.2f}, 副科={Qc_forecast[:, 1].sum():.2f}, "
      f"科级以下={Qc_forecast[:, 2].sum():.2f}, 总计={Qc_forecast.sum():.2f}")
print(f"    约束后：正科={Qc_forecast_adj[:, 0].sum():.2f}, 副科={Qc_forecast_adj[:, 1].sum():.2f}, "
      f"科级以下={Qc_forecast_adj[:, 2].sum():.2f}, 总计={Qc_forecast_adj.sum():.2f}")
print(f"    预算约束合计：{Budget_forecast.sum():.0f}")

# =================== 步骤6：测试集预算约束优化（新增） ===================
print(f"\n{'=' * 60}")
print(">>> 步骤6（补充）：测试集预算约束优化调整")
print(f"{'=' * 60}")

Qc_test_adj = np.zeros_like(Qc_test)
for i in range(n_test):
    Qc_test_adj[i] = budget_adjust(Qc_test[i], Budget_test[i])

print(f"--- 6.3 测试集约束前后对比 ---")
print(f"    约束前：正科={Qc_test[:, 0].sum():.2f}, 副科={Qc_test[:, 1].sum():.2f}, "
      f"科级以下={Qc_test[:, 2].sum():.2f}, 总计={Qc_test.sum():.2f}")
print(f"    约束后：正科={Qc_test_adj[:, 0].sum():.2f}, 副科={Qc_test_adj[:, 1].sum():.2f}, "
      f"科级以下={Qc_test_adj[:, 2].sum():.2f}, 总计={Qc_test_adj.sum():.2f}")
print(f"    预算约束合计：{Budget_test.sum():.0f}")

print(f"\n--- 6.4 测试集超预算部门 ---")
for i in range(n_test):
    total_before = Qc_test[i].sum()
    if total_before > Budget_test[i] + 0.01:
        print(f"    {df_test.iloc[i]['部门名称']}:")
        print(f"        预算={Budget_test[i]:.0f}, 约束前={total_before:.2f}, 约束后={Qc_test_adj[i].sum():.2f}")
        print(f"        拆分: 正科={Qc_test_adj[i,0]:.2f}, 副科={Qc_test_adj[i,1]:.2f}, 科以下={Qc_test_adj[i,2]:.2f}")

# =================== 步骤7：测试集假设检验 + 精度指标 ===================
print(f"\n{'=' * 60}")
print(">>> 步骤7：测试集假设检验与预测精度指标（无约束）")
print(f"{'=' * 60}")

test_metrics = {}
for j, col_name in enumerate(OUTPUT_COLS):
    actual = Output_test[:, j]
    predicted = Qc_test[:, j]
    residual = predicted - actual
    n = len(actual)

    print(f"\n--- 7.{j + 1} {col_name} ---")

    # 配对t检验
    t_stat, p_mean = stats.ttest_rel(predicted, actual)
    mean_diff = np.mean(residual)
    mean_conclusion = "通过" if p_mean > ALPHA else "未通过"
    print(f"    [总体均值] 配对t检验:")
    print(f"        均值差={mean_diff:.4f}, p={p_mean:.4f} → {mean_conclusion}")

    # F检验（原始方差，无缩尾）
    var_pred = np.var(predicted, ddof=1)
    var_actual = np.var(actual, ddof=1)
    if var_pred > var_actual:
        f_stat = var_pred / var_actual
        p_tail = 1 - stats.f.cdf(f_stat, n - 1, n - 1)
    else:
        f_stat = var_actual / var_pred
        p_tail = 1 - stats.f.cdf(f_stat, n - 1, n - 1)
    p_var_f = 2 * min(p_tail, 1 - p_tail)
    p_var_f = min(p_var_f, 1.0)
    f_conclusion = "通过" if p_var_f > ALPHA else "未通过"
    print(f"    [总体方差] F检验:")
    print(
        f"        预测方差={var_pred:.4f}, 实际方差={var_actual:.4f}, F={f_stat:.4f}, p={p_var_f:.4f} → {f_conclusion}")

    # Levene检验
    stat_lev, p_lev = stats.levene(predicted, actual)
    lev_conclusion = "通过" if p_lev > ALPHA else "未通过"
    print(f"    [总体方差] Levene检验:")
    print(f"        p={p_lev:.4f} → {lev_conclusion}")

    # 精度指标
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    print(f"    [预测精度] MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    test_metrics[col_name] = {
        'p_mean': round(p_mean, 4), 'mean_conclusion': mean_conclusion,
        'p_var_f': round(p_var_f, 4), 'f_conclusion': f_conclusion,
        'p_lev': round(p_lev, 4), 'lev_conclusion': lev_conclusion,
        'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4),
    }

# =================== 结果输出 ===================
print(f"\n{'=' * 60}")
print(">>> 最终结果输出")
print(f"{'=' * 60}")

test_summary = pd.DataFrame({
    '检验/指标': ['配对t检验(p值)', 'F检验(p值)', 'Levene检验(p值)', 'MAE', 'RMSE', 'R²'],
    '正科': [
        f"{test_metrics['正科']['p_mean']} ({test_metrics['正科']['mean_conclusion']})",
        f"{test_metrics['正科']['p_var_f']} ({test_metrics['正科']['f_conclusion']})",
        f"{test_metrics['正科']['p_lev']} ({test_metrics['正科']['lev_conclusion']})",
        test_metrics['正科']['mae'],
        test_metrics['正科']['rmse'],
        test_metrics['正科']['r2'],
    ],
    '副科': [
        f"{test_metrics['副科']['p_mean']} ({test_metrics['副科']['mean_conclusion']})",
        f"{test_metrics['副科']['p_var_f']} ({test_metrics['副科']['f_conclusion']})",
        f"{test_metrics['副科']['p_lev']} ({test_metrics['副科']['lev_conclusion']})",
        test_metrics['副科']['mae'],
        test_metrics['副科']['rmse'],
        test_metrics['副科']['r2'],
    ],
    '科级以下': [
        f"{test_metrics['科级以下']['p_mean']} ({test_metrics['科级以下']['mean_conclusion']})",
        f"{test_metrics['科级以下']['p_var_f']} ({test_metrics['科级以下']['f_conclusion']})",
        f"{test_metrics['科级以下']['p_lev']} ({test_metrics['科级以下']['lev_conclusion']})",
        test_metrics['科级以下']['mae'],
        test_metrics['科级以下']['rmse'],
        test_metrics['科级以下']['r2'],
    ]
})

print("\n=== 测试集假设检验与精度指标汇总 ===")
print(test_summary.to_string(index=False))

forecast_result = pd.DataFrame({
    '大类': df_forecast['大类'].values,
    '部门名称': df_forecast['部门名称'].values,
    '预算约束人数': Budget_forecast.astype(int),
    '约束前_正科': np.round(Qc_forecast[:, 0], 2),
    '约束前_副科': np.round(Qc_forecast[:, 1], 2),
    '约束前_科级以下': np.round(Qc_forecast[:, 2], 2),
    '约束前_合计': np.round(Qc_forecast.sum(axis=1), 2),
    '约束后_正科': np.round(Qc_forecast_adj[:, 0], 2),
    '约束后_副科': np.round(Qc_forecast_adj[:, 1], 2),
    '约束后_科级以下': np.round(Qc_forecast_adj[:, 2], 2),
    '约束后_合计': np.round(Qc_forecast_adj.sum(axis=1), 2),
})

forecast_summary = pd.DataFrame([{
    '大类': '合计',
    '部门名称': '-',
    '预算约束人数': int(Budget_forecast.sum()),
    '约束前_正科': np.round(Qc_forecast[:, 0].sum(), 2),
    '约束前_副科': np.round(Qc_forecast[:, 1].sum(), 2),
    '约束前_科级以下': np.round(Qc_forecast[:, 2].sum(), 2),
    '约束前_合计': np.round(Qc_forecast.sum(), 2),
    '约束后_正科': np.round(Qc_forecast_adj[:, 0].sum(), 2),
    '约束后_副科': np.round(Qc_forecast_adj[:, 1].sum(), 2),
    '约束后_科级以下': np.round(Qc_forecast_adj[:, 2].sum(), 2),
    '约束后_合计': np.round(Qc_forecast_adj.sum(), 2),
}])
forecast_result = pd.concat([forecast_result, forecast_summary], ignore_index=True)

print("\n=== 预测集约束前与约束后结果 ===")
print(forecast_result.to_string(index=False))

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


save_path_test = safe_save(test_summary, 'result_test_metrics.csv')
save_path_forecast = safe_save(forecast_result, 'result_forecast.csv')

print(f"\n{'=' * 60}")
print("[完成] 结果保存")
print(f"{'=' * 60}")
print(f"测试集指标已保存至：{save_path_test}")
print(f"预测集结果已保存至：{save_path_forecast}")
# =================== 测试集约束结果输出（新增） ===================
test_result = pd.DataFrame({
    '大类': df_test['大类'].values,
    '部门名称': df_test['部门名称'].values,
    '预算约束人数': Budget_test.astype(int),
    '约束前_正科': np.round(Qc_test[:, 0], 2),
    '约束前_副科': np.round(Qc_test[:, 1], 2),
    '约束前_科级以下': np.round(Qc_test[:, 2], 2),
    '约束前_合计': np.round(Qc_test.sum(axis=1), 2),
    '约束后_正科': np.round(Qc_test_adj[:, 0], 2),
    '约束后_副科': np.round(Qc_test_adj[:, 1], 2),
    '约束后_科级以下': np.round(Qc_test_adj[:, 2], 2),
    '约束后_合计': np.round(Qc_test_adj.sum(axis=1), 2),
})

test_summary_df = pd.DataFrame([{
    '大类': '合计',
    '部门名称': '-',
    '预算约束人数': int(Budget_test.sum()),
    '约束前_正科': np.round(Qc_test[:, 0].sum(), 2),
    '约束前_副科': np.round(Qc_test[:, 1].sum(), 2),
    '约束前_科级以下': np.round(Qc_test[:, 2].sum(), 2),
    '约束前_合计': np.round(Qc_test.sum(), 2),
    '约束后_正科': np.round(Qc_test_adj[:, 0].sum(), 2),
    '约束后_副科': np.round(Qc_test_adj[:, 1].sum(), 2),
    '约束后_科级以下': np.round(Qc_test_adj[:, 2].sum(), 2),
    '约束后_合计': np.round(Qc_test_adj.sum(), 2),
}])
test_result = pd.concat([test_result, test_summary_df], ignore_index=True)

print("\n=== 测试集约束前与约束后结果 ===")
print(test_result.to_string(index=False))

save_path_test_result = safe_save(test_result, 'result_test_forecast.csv')
print(f"测试集约束结果已保存至：{save_path_test_result}")