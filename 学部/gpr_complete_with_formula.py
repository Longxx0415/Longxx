import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # 修复 PyCharm 后端兼容问题
import matplotlib.pyplot as plt
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_2025_PATH = os.path.join(BASE_DIR, 'data2025.csv')
TRAIN_2026_PATH = os.path.join(BASE_DIR, 'data2026.csv')
PREDICT_2030_PATH = os.path.join(BASE_DIR, 'forecastdata.csv')
OFFICIAL_OLS_CSV = os.path.join(BASE_DIR, '测算数据（两个参数）.csv')
OFFICIAL_OLS_XLSX = os.path.join(BASE_DIR, '测算数据（两个参数）.xlsx')

EXCLUDE_NAMES = ['上海国际知识产权学院', '中德工程学院，职业技术教育学院']
FEATURE_COLS = ['本科生', '研究生', '在编-教师', '留学生人数', '公共课学分']

# 核函数配置
KERNEL = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

RATIOS = np.array([0.2, 0.3, 0.5])
ALPHA = 0.05

# ==================== 工具函数 ====================
def load_data(path, exclude_names):
    df = pd.read_csv(path)
    mask = ~df['部门名称'].isin(exclude_names)
    return df[mask].reset_index(drop=True)

def match_official_ols(dept_names, df_official):
    """匹配官方OLS拟合值，支持模糊匹配"""
    official_vals = []
    for name in dept_names:
        match = df_official[df_official['学院'] == name]
        if len(match) > 0:
            official_vals.append(match['拟合数'].values[0])
            continue
        for idx, off_name in enumerate(df_official['学院']):
            if name in off_name or off_name in name:
                official_vals.append(df_official.iloc[idx]['拟合数'])
                break
        else:
            official_vals.append(np.nan)
    return np.array(official_vals)

def budget_adjust(total_pred, B_budget):
    total_pred = float(total_pred)
    if total_pred <= B_budget + 1e-9:
        return total_pred, total_pred * RATIOS
    return B_budget, B_budget * RATIOS

def evaluate_metrics(y_pred, y_actual):
    n = len(y_actual)
    residual = y_pred - y_actual
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    mape = np.mean(np.abs(residual / y_actual)) * 100
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    t_stat, p_mean = stats.ttest_rel(y_pred, y_actual)
    mean_conclusion = "通过" if p_mean > ALPHA else "未通过"

    var_pred = np.var(y_pred, ddof=1)
    var_actual = np.var(y_actual, ddof=1)
    if var_pred > var_actual:
        f_stat = var_pred / var_actual
        p_tail = 1 - stats.f.cdf(f_stat, n-1, n-1)
    else:
        f_stat = var_actual / var_pred
        p_tail = 1 - stats.f.cdf(f_stat, n-1, n-1)
    p_var_f = 2 * min(p_tail, 1-p_tail)
    p_var_f = min(p_var_f, 1.0)
    f_conclusion = "通过" if p_var_f > ALPHA else "未通过"

    stat_lev, p_lev = stats.levene(y_pred, y_actual)
    lev_conclusion = "通过" if p_lev > ALPHA else "未通过"

    return {
        'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
        'p_mean': p_mean, 'mean_conclusion': mean_conclusion,
        'p_var_f': p_var_f, 'f_conclusion': f_conclusion,
        'p_lev': p_lev, 'lev_conclusion': lev_conclusion,
        't_stat': t_stat, 'f_stat': f_stat, 'stat_lev': stat_lev
    }

# ==================== 模式1：25→26 验证（假设检验与精度）====================
print("="*80)
print(">>> 模式一：2025年训练 → 2026年验证（假设检验与精度指标）")
print("="*80)

df_train = load_data(TRAIN_2025_PATH, EXCLUDE_NAMES)
df_val = load_data(TRAIN_2026_PATH, EXCLUDE_NAMES)

dept_names = df_train['部门名称'].tolist()
df_val = df_val.set_index('部门名称').reindex(dept_names).reset_index()

X_train = df_train[FEATURE_COLS].values.astype(float)
y_train = df_train['编制人数'].values.astype(float)
X_val = df_val[FEATURE_COLS].values.astype(float)
y_actual = df_val['编制人数'].values.astype(float)
budget = df_val['预算约束人数'].values.astype(float)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

gpr = GaussianProcessRegressor(
    kernel=KERNEL, n_restarts_optimizer=50, normalize_y=True,
    alpha=1e-2, random_state=RANDOM_STATE
)
gpr.fit(X_train_scaled, y_train)

y_pred, y_std = gpr.predict(X_val_scaled, return_std=True)
y_pred = np.maximum(y_pred, 0)

metrics = evaluate_metrics(y_pred, y_actual)

print(f"\n--- 假设检验结果 ---")
print(f"配对t检验: t={metrics['t_stat']:.4f}, p={metrics['p_mean']:.4f} → {metrics['mean_conclusion']}")
print(f"F检验: F={metrics['f_stat']:.4f}, p={metrics['p_var_f']:.4f} → {metrics['f_conclusion']}")
print(f"Levene检验: p={metrics['p_lev']:.4f} → {metrics['lev_conclusion']}")
print(f"\n--- 精度指标 ---")
print(f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")

# 预算约束与层级拆分
n_depts = len(dept_names)
adj_total = np.zeros(n_depts)
adj_split = np.zeros((n_depts, 3))
for i in range(n_depts):
    adj_total[i], adj_split[i] = budget_adjust(y_pred[i], budget[i])

print(f"\n--- 预算约束处理（2:3:5拆分）---")
print(f"约束前总编制: {y_pred.sum():.2f}")
print(f"约束后总编制: {adj_total.sum():.2f}")
print(f"预算约束合计: {budget.sum():.0f}")
print(f"超预算学院数: {np.sum(y_pred > budget + 0.01)} / {n_depts}")

# 输出详细结果表
result_val = pd.DataFrame({
    '部门名称': dept_names,
    '实际编制': y_actual.astype(int),
    'GPR预测': np.round(y_pred, 2),
    'GPR_std': np.round(y_std, 2),
    '预算约束': budget.astype(int),
    '约束后总编': np.round(adj_total, 2),
    '正科(20%)': np.round(adj_split[:, 0], 2),
    '副科(30%)': np.round(adj_split[:, 1], 2),
    '科级以下(50%)': np.round(adj_split[:, 2], 2),
    '偏差': np.round(y_pred - y_actual, 2),
})

summary = pd.DataFrame([{
    '部门名称': '合计',
    '实际编制': int(y_actual.sum()),
    'GPR预测': round(y_pred.sum(), 2),
    'GPR_std': '-',
    '预算约束': int(budget.sum()),
    '约束后总编': round(adj_total.sum(), 2),
    '正科(20%)': round(adj_split[:, 0].sum(), 2),
    '副科(30%)': round(adj_split[:, 1].sum(), 2),
    '科级以下(50%)': round(adj_split[:, 2].sum(), 2),
    '偏差': round(y_pred.sum() - y_actual.sum(), 2),
}])
result_val = pd.concat([result_val, summary], ignore_index=True)

print(f"\n--- 2026年验证结果表 ---")
print(result_val.to_string(index=False))

result_val.to_csv(os.path.join(BASE_DIR, 'gpr_validation_2026.csv'), index=False, encoding='utf-8-sig')
print(f"\n[保存] gpr_validation_2026.csv")

# ==================== 模式2：26→30 预测 + 官方OLS对比图 ====================
print("\n" + "="*80)
print(">>> 模式二：2026年训练 → 2030年预测（与官方OLS拟合对比）")
print("="*80)

df_train_26 = load_data(TRAIN_2026_PATH, EXCLUDE_NAMES)
df_pred_30 = load_data(PREDICT_2030_PATH, EXCLUDE_NAMES)

dept_names_30 = df_train_26['部门名称'].tolist()
df_pred_30 = df_pred_30.set_index('部门名称').reindex(dept_names_30).reset_index()

X_train_26 = df_train_26[FEATURE_COLS].values.astype(float)
y_train_26 = df_train_26['编制人数'].values.astype(float)
X_pred_30 = df_pred_30[FEATURE_COLS].values.astype(float)

scaler_26 = StandardScaler()
X_train_26_scaled = scaler_26.fit_transform(X_train_26)
X_pred_30_scaled = scaler_26.transform(X_pred_30)

gpr_30 = GaussianProcessRegressor(
    kernel=KERNEL, n_restarts_optimizer=50, normalize_y=True,
    alpha=1e-2, random_state=RANDOM_STATE
)
gpr_30.fit(X_train_26_scaled, y_train_26)

kernel_trained = gpr_30.kernel_
sigma2 = kernel_trained.k1.k1.constant_value
l = kernel_trained.k1.k2.length_scale
sigma_n2 = kernel_trained.k2.noise_level

print(f"\n--- GPR核函数参数 ---")
print(f"信号方差 σ² = {sigma2:.6f}")
print(f"长度尺度 l = {l:.6f}")
print(f"噪声方差 σ_n² = {sigma_n2:.6f}")

y_pred_30, y_std_30 = gpr_30.predict(X_pred_30_scaled, return_std=True)
y_pred_30 = np.maximum(y_pred_30, 0)

budget_30 = df_pred_30['预算约束人数'].values.astype(float)

adj_total_30 = np.zeros(n_depts)
adj_split_30 = np.zeros((n_depts, 3))
for i in range(n_depts):
    adj_total_30[i], adj_split_30[i] = budget_adjust(y_pred_30[i], budget_30[i])

print(f"\n--- 2030年预测结果汇总 ---")
print(f"GPR预测合计（约束前）: {y_pred_30.sum():.2f}")
print(f"GPR预测合计（约束后）: {adj_total_30.sum():.2f}")
print(f"预算约束合计: {budget_30.sum():.0f}")
print(f"超预算学院数: {np.sum(y_pred_30 > budget_30 + 0.01)} / {n_depts}")

over_budget = []
for i in range(n_depts):
    if y_pred_30[i] > budget_30[i] + 0.01:
        over_budget.append((dept_names_30[i], y_pred_30[i], budget_30[i]))
print(f"\n超预算学院:")
for name, pred, bud in over_budget:
    print(f"  {name}: 预测{pred:.2f} > 预算{bud:.0f}")

# 结果表
result_30 = pd.DataFrame({
    '部门名称': dept_names_30,
    'GPR预测': np.round(y_pred_30, 2),
    'GPR_std': np.round(y_std_30, 2),
    '95%CI下限': np.round(y_pred_30 - 1.96*y_std_30, 2),
    '95%CI上限': np.round(y_pred_30 + 1.96*y_std_30, 2),
    '预算约束': budget_30.astype(int),
    '约束后总编': np.round(adj_total_30, 2),
    '正科(20%)': np.round(adj_split_30[:, 0], 2),
    '副科(30%)': np.round(adj_split_30[:, 1], 2),
    '科级以下(50%)': np.round(adj_split_30[:, 2], 2),
})

summary_30 = pd.DataFrame([{
    '部门名称': '合计',
    'GPR预测': round(y_pred_30.sum(), 2),
    'GPR_std': '-',
    '95%CI下限': '-',
    '95%CI上限': '-',
    '预算约束': int(budget_30.sum()),
    '约束后总编': round(adj_total_30.sum(), 2),
    '正科(20%)': round(adj_split_30[:, 0].sum(), 2),
    '副科(30%)': round(adj_split_30[:, 1].sum(), 2),
    '科级以下(50%)': round(adj_split_30[:, 2].sum(), 2),
}])
result_30 = pd.concat([result_30, summary_30], ignore_index=True)

print(f"\n--- 2030年预测结果表 ---")
print(result_30.to_string(index=False))

result_30.to_csv(os.path.join(BASE_DIR, 'gpr_forecast_2030.csv'), index=False, encoding='utf-8-sig')
print(f"\n[保存] gpr_forecast_2030.csv")

# ==================== 官方OLS对比图 ====================
print(f"\n{'='*80}")
print(">>> 生成 GPR预测 vs 官方OLS 对比图")
print(f"{'='*80}")

official = None
try:
    if os.path.exists(OFFICIAL_OLS_CSV):
        df_official = pd.read_csv(OFFICIAL_OLS_CSV)
        official = match_official_ols(dept_names_30, df_official)
        print(f"[读取] 官方OLS CSV: {OFFICIAL_OLS_CSV}")
    elif os.path.exists(OFFICIAL_OLS_XLSX):
        df_official = pd.read_excel(OFFICIAL_OLS_XLSX, sheet_name='不含组织员')
        official = match_official_ols(dept_names_30, df_official)
        print(f"[读取] 官方OLS XLSX: {OFFICIAL_OLS_XLSX}")
    else:
        print(f"[警告] 未找到官方OLS文件，跳过对比图生成")
        print(f"  请确保以下文件之一存在于同目录:")
        print(f"  - {OFFICIAL_OLS_CSV}")
        print(f"  - {OFFICIAL_OLS_XLSX}")
except Exception as e:
    print(f"[错误] 读取官方OLS文件失败: {e}")

if official is not None and not np.all(np.isnan(official)):
    # 计算与官方OLS的精度
    mae_ols = np.mean(np.abs(y_pred_30 - official))
    rmse_ols = np.sqrt(np.mean((y_pred_30 - official)**2))
    ss_res_ols = np.sum((y_pred_30 - official)**2)
    ss_tot_ols = np.sum((official - np.mean(official))**2)
    r2_ols = 1 - ss_res_ols / ss_tot_ols if ss_tot_ols != 0 else 0.0

    print(f"\n--- 与官方OLS拟合值对比 ---")
    print(f"MAE={mae_ols:.4f}, RMSE={rmse_ols:.4f}, R²={r2_ols:.4f}")

    # 打印对比表
    comp_df = pd.DataFrame({
        '部门名称': dept_names_30,
        '官方OLS': np.round(official, 2),
        'GPR预测': np.round(y_pred_30, 2),
        '偏差': np.round(y_pred_30 - official, 2),
    })
    print(comp_df.to_string(index=False))
    comp_df.to_csv(os.path.join(BASE_DIR, 'gpr_vs_ols_2030.csv'), index=False, encoding='utf-8-sig')

    # 生成对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    short_names = []
    for name in dept_names_30:
        if '（' in name: short = name.split('（')[0]
        elif '，' in name: short = name.split('，')[0]
        else: short = name[:6]
        short_names.append(short)

    x = np.arange(len(dept_names_30))

    # 图1: 柱状对比
    ax1 = axes[0]
    width = 0.35
    ax1.bar(x - width/2, official, width, label='官方OLS', color='gray', alpha=0.7)
    ax1.bar(x + width/2, y_pred_30, width, label='GPR预测', color='#e74c3c', alpha=0.8)
    ax1.errorbar(x + width/2, y_pred_30, yerr=1.96*y_std_30, fmt='none', color='black', alpha=0.3, capsize=2)
    ax1.set_ylabel('编制人数')
    ax1.set_title(f'GPR预测 vs 官方OLS (R²={r2_ols:.3f}, MAE={mae_ols:.2f})', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 图2: 散点图
    ax2 = axes[1]
    max_val = max(official.max(), y_pred_30.max()) * 1.1
    ax2.scatter(official, y_pred_30, c='#e74c3c', alpha=0.7, s=80, edgecolors='white', zorder=3)
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x', zorder=1)
    ax2.set_xlabel('官方OLS拟合值')
    ax2.set_ylabel('GPR预测值')
    ax2.set_title(f'散点图 (R²={r2_ols:.3f})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)

    plt.suptitle('GPR 2030年预测 vs 官方OLS拟合对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'gpr_vs_ols_2030.png'), dpi=200, bbox_inches='tight')
    print(f"\n[保存] gpr_vs_ols_2030.png")
else:
    print(f"\n[跳过] 未找到官方OLS数据，无法生成对比图")
    print(f"  请准备官方OLS文件后重新运行")

print(f"\n{'='*80}")
print("[完成] 所有任务执行完毕")
print(f"{'='*80}")