import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
# 数据文件路径（按年份）
DATA_PATHS = {
    2022: 'data2022.csv',
    2023: 'data2023.csv',
    2024: 'data2024.csv',
    2025: 'data2025.csv',
    2026: 'data2026.csv',
}
FORECAST_PATH = 'forecastdata.csv'

# Q1特征：本科生、研究生、在编教师（已去掉党员人数）
Q1_COLS = ['本科生', '研究生', '在编-教师']
N_COMPONENTS = 3  # 主成分保留数改为2
RANDOM_STATE = 42

# 拆分比例：正科:副科:科级以下 = 2:3:5
RATIOS = np.array([0.2, 0.3, 0.5])
RATIO_NAMES = ['正科(20%)', '副科(30%)', '科级以下(50%)']
OUTPUT_COLS = ['正科', '副科', '科级以下']

ALPHA = 0.05
# ================================================


def load_data():
    """读取所有年份数据"""
    dfs = {}
    for year, path in DATA_PATHS.items():
        dfs[year] = pd.read_csv(path)
    df_forecast = pd.read_csv(FORECAST_PATH)
    return dfs, df_forecast


def build_q1_model(df_2026):
    """
    Q1模型构建：仅用2026年数据拟合PCA，计算每个学院各自的职能强度指数h
    """
    scaler = StandardScaler()
    input1 = df_2026[Q1_COLS].values.astype(float)
    input1_scaled = scaler.fit_transform(input1)

    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    pca.fit(input1_scaled)
    components = pca.components_
    var_ratio = pca.explained_variance_ratio_

    def compute_H(df):
        input1_new = df[Q1_COLS].values.astype(float)
        scores = input1_new @ components.T  # ← 原始数据直接乘，不加 scaler.transform
        H = scores @ var_ratio
        return H

    # 计算2026年的H和h（每个学院各自保留）
    H_2026 = compute_H(df_2026)
    actual_2026 = df_2026['编制人数'].values.astype(float)
    h_per_dept = H_2026 / actual_2026
    # 保护：避免除零
    h_per_dept = np.where(np.abs(h_per_dept) < 1e-10, 1e-10, h_per_dept)

    def q1_predict(df):
        H = compute_H(df)
        pred = H / h_per_dept
        pred = np.maximum(pred, 0)  # 编制人数不能为负
        return pred

    return q1_predict, scaler, pca, h_per_dept, components, var_ratio


def q2_exp_smooth_predict(history_dfs, target_year_offset, dept_idx):
    """
    Q2指数平滑预测：对每个学院的历史编制人数序列进行预测
    history_dfs: 历史年份df列表（按时间顺序）
    target_year_offset: 从最后一个历史年份到目标年份的步数
    dept_idx: 学院索引
    """
    hist = [float(df['编制人数'].iloc[dept_idx]) for df in history_dfs]
    n = len(hist)
    if n == 0:
        return np.nan
    if n == 1:
        return hist[0]

    # 优先使用Holt线性趋势法
    try:
        model = Holt(hist, exponential=False)
        fit = model.fit(optimized=True)
        pred = fit.forecast(target_year_offset)
        return float(pred[-1] if target_year_offset > 1 else pred[0])
    except Exception:
        # 回退到简单指数平滑
        try:
            model = SimpleExpSmoothing(hist)
            fit = model.fit(optimized=True)
            pred = fit.forecast(target_year_offset)
            return float(pred[-1] if target_year_offset > 1 else pred[0])
        except Exception:
            # 再回退到线性外推
            if n >= 2:
                slope = (hist[-1] - hist[0]) / (n - 1)
                return float(hist[-1] + slope * target_year_offset)
            return hist[-1]


def budget_adjust(total_pred, B, ratios=RATIOS):
    """
    预算约束：总编制超预算时压缩到预算上限，再按比例拆分
    """
    total_pred = float(total_pred)
    if total_pred <= B + 1e-9:
        return total_pred * ratios
    return B * ratios


def split_total(total, ratios=RATIOS):
    """按给定比例拆分总编制为三个层级"""
    return total * ratios


# =================== 读取数据 ===================
dfs, df_forecast = load_data()
df_2026 = dfs[2026]
n_depts = len(df_2026)
dept_names = df_2026['部门名称'].tolist()
dept_cats = df_2026['大类'].tolist()

print(f"{'='*70}")
print(">>> 数据读取与配置")
print(f"{'='*70}")
print(f"学院数量: {n_depts}")
print(f"Q1特征（PCA）: {Q1_COLS}")
print(f"主成分保留数: {N_COMPONENTS}")
print(f"预测目标: 编制人数（最终按2:3:5拆分）")
print(f"Q2方法: 指数平滑法（Holt线性趋势）")
print(f"组合定权: 滚动窗口验证，各学院独立权重")

# =================== Q1模型构建（仅用2026年） ===================
print(f"\n{'='*70}")
print(">>> Q1: PCA工作量模型构建（仅用2026年数据）")
print(f"{'='*70}")

q1_predict, scaler_q1, pca_model, h_per_dept, components, var_ratio = build_q1_model(df_2026)

print(f"\n--- 1.1 PCA拟合结果 ---")
print(f"    保留主成分: {N_COMPONENTS}个，原始特征: {len(Q1_COLS)}个")
print(f"    方差贡献率: {np.round(var_ratio, 4)}")
print(f"    累计贡献率: {np.round(var_ratio.sum(), 4)}")

loadings_df = pd.DataFrame(
    components.T,
    index=Q1_COLS,
    columns=[f'PC{k+1}' for k in range(N_COMPONENTS)]
)
print(f"\n--- 1.2 主成分载荷矩阵 ---")
print(loadings_df.round(4).to_string())

print(f"\n--- 1.3 各学院职能强度指数h ---")
for i, name in enumerate(dept_names):
    print(f"    {name}: h={h_per_dept[i]:.6f}")

# =================== 滚动窗口验证（定权） ===================
print(f"\n{'='*70}")
print(">>> 滚动窗口验证：各学院组合权重计算")
print(f"{'='*70}")

# 三个验证窗口
windows = [
    ("22-23 → 24", [dfs[2022], dfs[2023]], dfs[2024], 1),
    ("22-23-24 → 25", [dfs[2022], dfs[2023], dfs[2024]], dfs[2025], 1),
    ("22-23-24-25 → 26", [dfs[2022], dfs[2023], dfs[2024], dfs[2025]], dfs[2026], 1),
]

errors_q1 = np.zeros((n_depts, 3))
errors_q2 = np.zeros((n_depts, 3))

for w_idx, (desc, hist_dfs, actual_df, offset) in enumerate(windows):
    print(f"\n--- 窗口 {w_idx+1}: {desc} ---")
    q1_pred = q1_predict(actual_df)
    q2_pred = np.array([q2_exp_smooth_predict(hist_dfs, offset, i) for i in range(n_depts)])
    actual = actual_df['编制人数'].values.astype(float)

    errors_q1[:, w_idx] = (q1_pred - actual) ** 2
    errors_q2[:, w_idx] = (q2_pred - actual) ** 2

    mae_q1 = np.mean(np.abs(q1_pred - actual))
    mae_q2 = np.mean(np.abs(q2_pred - actual))
    print(f"    Q1 MAE={mae_q1:.4f}, Q2 MAE={mae_q2:.4f}")

# 各学院误差方差（用MSE）
var_q1 = np.mean(errors_q1, axis=1)
var_q2 = np.mean(errors_q2, axis=1)
var_q1 = np.where(var_q1 < 1e-10, 1e-10, var_q1)
var_q2 = np.where(var_q2 < 1e-10, 1e-10, var_q2)

# 方差倒数定权
w_q1 = (1 / var_q1) / (1 / var_q1 + 1 / var_q2)
w_q2 = (1 / var_q2) / (1 / var_q1 + 1 / var_q2)

print(f"\n--- 各学院组合权重（方差倒数法）---")
for i, name in enumerate(dept_names):
    print(f"    {name}: w_Q1={w_q1[i]:.4f}, w_Q2={w_q2[i]:.4f}")

# =================== 假设检验：用22-23-24-25预测26年 ===================
print(f"\n{'='*70}")
print(">>> 假设检验与预测精度（验证26年，未约束）")
print(f"{'='*70}")

q1_26 = q1_predict(df_2026)
q2_26 = np.array([q2_exp_smooth_predict([dfs[2022], dfs[2023], dfs[2024], dfs[2025]], 1, i)
                  for i in range(n_depts)])
qc_26 = w_q1 * q1_26 + w_q2 * q2_26
qc_26 = np.maximum(qc_26, 0)  # 截断保护
actual_26 = df_2026['编制人数'].values.astype(float)

residual_26 = qc_26 - actual_26
n = n_depts

print(f"\n--- 实际值 vs 预测值统计 ---")
print(f"    实际值: 均值={actual_26.mean():.2f}, 方差={np.var(actual_26, ddof=1):.4f}")
print(f"    预测值: 均值={qc_26.mean():.2f}, 方差={np.var(qc_26, ddof=1):.4f}")

# 配对t检验
t_stat, p_mean = stats.ttest_rel(qc_26, actual_26)
mean_diff = np.mean(residual_26)
mean_conclusion = "通过" if p_mean > ALPHA else "未通过"
print(f"\n    [总体均值] 配对t检验:")
print(f"        均值差={mean_diff:.4f}, p={p_mean:.4f} → {mean_conclusion}")

# F检验
var_pred = np.var(qc_26, ddof=1)
var_actual = np.var(actual_26, ddof=1)
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
print(f"        F={f_stat:.4f}, p={p_var_f:.4f} → {f_conclusion}")

# Levene检验
stat_lev, p_lev = stats.levene(qc_26, actual_26)
lev_conclusion = "通过" if p_lev > ALPHA else "未通过"
print(f"    [总体方差] Levene检验:")
print(f"        p={p_lev:.4f} → {lev_conclusion}")

# 精度指标
mae = np.mean(np.abs(residual_26))
rmse = np.sqrt(np.mean(residual_26**2))
ss_res = np.sum(residual_26**2)
ss_tot = np.sum((actual_26 - np.mean(actual_26))**2)
r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
print(f"    [预测精度] MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# 保存指标
test_metrics = {
    'p_mean': round(p_mean, 4), 'mean_conclusion': mean_conclusion,
    'p_var_f': round(p_var_f, 4), 'f_conclusion': f_conclusion,
    'p_lev': round(p_lev, 4), 'lev_conclusion': lev_conclusion,
    'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4),
}

# =================== 26年约束后结果（参考） ===================
print(f"\n{'='*70}")
print(">>> 26年验证集约束后结果（参考）")
print(f"{'='*70}")

budget_26 = df_2026['预算约束人数'].values.astype(float)
qc_26_split = np.zeros((n_depts, 3))
qc_26_adj_total = np.zeros(n_depts)
qc_26_adj_split = np.zeros((n_depts, 3))

for i in range(n_depts):
    qc_26_split[i] = split_total(qc_26[i])
    qc_26_adj_split[i] = budget_adjust(qc_26[i], budget_26[i])
    qc_26_adj_total[i] = qc_26_adj_split[i].sum()

print(f"\n--- 26年约束前后对比 ---")
for i in range(n_depts):
    if qc_26[i] > budget_26[i] + 0.01:
        print(f"    {dept_names[i]}:")
        print(f"        预算={budget_26[i]:.0f}, 约束前={qc_26[i]:.2f}, 约束后={qc_26_adj_total[i]:.2f}")
        print(f"        拆分: 正科={qc_26_adj_split[i,0]:.2f}, 副科={qc_26_adj_split[i,1]:.2f}, 科以下={qc_26_adj_split[i,2]:.2f}")

print(f"\n    约束前总编制合计: {qc_26.sum():.2f}")
print(f"    约束后总编制合计: {qc_26_adj_total.sum():.2f}")
print(f"    预算约束合计: {budget_26.sum():.0f}")

# =================== 最终预测：30年 ===================
print(f"\n{'='*70}")
print(">>> 最终预测：2030年编制人数")
print(f"{'='*70}")

q1_30 = q1_predict(df_forecast)
q2_30 = np.array([q2_exp_smooth_predict([dfs[2022], dfs[2023], dfs[2024], dfs[2025], dfs[2026]], 4, i)
                  for i in range(n_depts)])
qc_30 = w_q1 * q1_30 + w_q2 * q2_30
qc_30 = np.maximum(qc_30, 0)

budget_30 = df_forecast['预算约束人数'].values.astype(float)

# 约束前拆分
qc_30_split = np.zeros((n_depts, 3))
for i in range(n_depts):
    qc_30_split[i] = split_total(qc_30[i])

# 约束后拆分
qc_30_adj_total = np.zeros(n_depts)
qc_30_adj_split = np.zeros((n_depts, 3))
for i in range(n_depts):
    qc_30_adj_split[i] = budget_adjust(qc_30[i], budget_30[i])
    qc_30_adj_total[i] = qc_30_adj_split[i].sum()

print(f"\n--- 30年预测结果汇总 ---")
print(f"    Q1预测合计: {q1_30.sum():.2f}")
print(f"    Q2预测合计: {q2_30.sum():.2f}")
print(f"    组合预测合计（约束前）: {qc_30.sum():.2f}")
print(f"    组合预测合计（约束后）: {qc_30_adj_total.sum():.2f}")
print(f"    预算约束合计: {budget_30.sum():.0f}")

print(f"\n--- 30年超预算部门 ---")
for i in range(n_depts):
    if qc_30[i] > budget_30[i] + 0.01:
        print(f"    {dept_names[i]}:")
        print(f"        预算={budget_30[i]:.0f}, 约束前={qc_30[i]:.2f}, 约束后={qc_30_adj_total[i]:.2f}")
        print(f"        拆分: 正科={qc_30_adj_split[i,0]:.2f}, 副科={qc_30_adj_split[i,1]:.2f}, 科以下={qc_30_adj_split[i,2]:.2f}")

# =================== 结果输出表格 ===================
print(f"\n{'='*70}")
print(">>> 最终结果输出")
print(f"{'='*70}")

# 1. 假设检验与精度指标汇总表
test_summary = pd.DataFrame({
    '检验/指标': ['配对t检验(p值)', 'F检验(p值)', 'Levene检验(p值)', 'MAE', 'RMSE', 'R²'],
    '验证26年_总编制': [
        f"{test_metrics['p_mean']} ({test_metrics['mean_conclusion']})",
        f"{test_metrics['p_var_f']} ({test_metrics['f_conclusion']})",
        f"{test_metrics['p_lev']} ({test_metrics['lev_conclusion']})",
        test_metrics['mae'],
        test_metrics['rmse'],
        test_metrics['r2'],
    ]
})
print("\n=== 验证26年假设检验与精度指标汇总 ===")
print(test_summary.to_string(index=False))

# 2. 26年约束前后结果表
result_26 = pd.DataFrame({
    '大类': dept_cats,
    '部门名称': dept_names,
    '预算约束人数': budget_26.astype(int),
    '约束前_总编制': np.round(qc_26, 2),
    '约束前_正科': np.round(qc_26_split[:, 0], 2),
    '约束前_副科': np.round(qc_26_split[:, 1], 2),
    '约束前_科级以下': np.round(qc_26_split[:, 2], 2),
    '约束后_总编制': np.round(qc_26_adj_total, 2),
    '约束后_正科': np.round(qc_26_adj_split[:, 0], 2),
    '约束后_副科': np.round(qc_26_adj_split[:, 1], 2),
    '约束后_科级以下': np.round(qc_26_adj_split[:, 2], 2),
})
result_26_summary = pd.DataFrame([{
    '大类': '合计',
    '部门名称': '-',
    '预算约束人数': int(budget_26.sum()),
    '约束前_总编制': np.round(qc_26.sum(), 2),
    '约束前_正科': np.round(qc_26_split[:, 0].sum(), 2),
    '约束前_副科': np.round(qc_26_split[:, 1].sum(), 2),
    '约束前_科级以下': np.round(qc_26_split[:, 2].sum(), 2),
    '约束后_总编制': np.round(qc_26_adj_total.sum(), 2),
    '约束后_正科': np.round(qc_26_adj_split[:, 0].sum(), 2),
    '约束后_副科': np.round(qc_26_adj_split[:, 1].sum(), 2),
    '约束后_科级以下': np.round(qc_26_adj_split[:, 2].sum(), 2),
}])
result_26 = pd.concat([result_26, result_26_summary], ignore_index=True)

print("\n=== 26年验证集约束前与约束后结果 ===")
print(result_26.to_string(index=False))

# 3. 30年预测结果表（约束前后）
result_30 = pd.DataFrame({
    '大类': dept_cats,
    '部门名称': dept_names,
    '预算约束人数': budget_30.astype(int),
    '约束前_总编制': np.round(qc_30, 2),
    '约束前_正科': np.round(qc_30_split[:, 0], 2),
    '约束前_副科': np.round(qc_30_split[:, 1], 2),
    '约束前_科级以下': np.round(qc_30_split[:, 2], 2),
    '约束后_总编制': np.round(qc_30_adj_total, 2),
    '约束后_正科': np.round(qc_30_adj_split[:, 0], 2),
    '约束后_副科': np.round(qc_30_adj_split[:, 1], 2),
    '约束后_科级以下': np.round(qc_30_adj_split[:, 2], 2),
})
result_30_summary = pd.DataFrame([{
    '大类': '合计',
    '部门名称': '-',
    '预算约束人数': int(budget_30.sum()),
    '约束前_总编制': np.round(qc_30.sum(), 2),
    '约束前_正科': np.round(qc_30_split[:, 0].sum(), 2),
    '约束前_副科': np.round(qc_30_split[:, 1].sum(), 2),
    '约束前_科级以下': np.round(qc_30_split[:, 2].sum(), 2),
    '约束后_总编制': np.round(qc_30_adj_total.sum(), 2),
    '约束后_正科': np.round(qc_30_adj_split[:, 0].sum(), 2),
    '约束后_副科': np.round(qc_30_adj_split[:, 1].sum(), 2),
    '约束后_科级以下': np.round(qc_30_adj_split[:, 2].sum(), 2),
}])
result_30 = pd.concat([result_30, result_30_summary], ignore_index=True)

print("\n=== 2030年预测集约束前与约束后结果 ===")
print(result_30.to_string(index=False))

# =================== 保存结果 ===================
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

save_path_metrics = safe_save(test_summary, 'result_2030_metrics.csv')
save_path_26 = safe_save(result_26, 'result_2026_validation.csv')
save_path_30 = safe_save(result_30, 'result_2030_forecast.csv')

print(f"\n{'='*70}")
print("[完成] 结果保存")
print(f"{'='*70}")
print(f"精度指标已保存至: {save_path_metrics}")
print(f"26年验证结果已保存至: {save_path_26}")
print(f"30年预测结果已保存至: {save_path_30}")