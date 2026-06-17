import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
DATA_PATHS = {
    2022: 'data2022.csv',
    2023: 'data2023.csv',
    2024: 'data2024.csv',
    2025: 'data2025.csv',
    2026: 'data2026.csv',
}
Q1_COLS = ['本科生', '研究生', '留学生人数', '在编-教师', '公共课学分']
N_COMPONENTS = 4
RANDOM_STATE = 42
RATIOS = np.array([0.2, 0.3, 0.5])  # 正科:副科:科级以下 = 2:3:5
ALPHA = 0.05
EXCLUDE_NAMES = ['上海国际知识产权学院', '中德工程学院，职业技术教育学院']
# ================================================


def load_and_preprocess():
    dfs = {}
    available_years = []
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    for year, path in DATA_PATHS.items():
        full_path = os.path.join(base_dir, path)
        if not os.path.exists(full_path):
            if os.path.exists(path):
                full_path = path
            else:
                continue
        df = pd.read_csv(full_path)
        # 对早期缺少的列填充0
        for col in Q1_COLS:
            if col not in df.columns:
                df[col] = 0
        mask = ~df['部门名称'].isin(EXCLUDE_NAMES)
        dfs[year] = df[mask].reset_index(drop=True)
        available_years.append(year)

    if not dfs:
        raise FileNotFoundError("未找到任何数据文件！")

    base_year = 2026 if 2026 in dfs else max(dfs.keys())
    dept_names = dfs[base_year]['部门名称'].tolist()
    dept_cats = dfs[base_year]['大类'].tolist()
    for year in dfs:
        dfs[year] = dfs[year].set_index('部门名称').reindex(dept_names).reset_index()
    return dfs, dept_names, dept_cats, available_years


def compute_actual_ratios(dfs):
    actual_ratios = {}
    actual_totals = {}
    for year in dfs.keys():
        total = dfs[year]['编制人数'].sum()
        actual_totals[year] = total
        actual_ratios[year] = dfs[year]['编制人数'].values.astype(float) / total
    return actual_ratios, actual_totals


def build_q1_model(df_train):
    scaler = StandardScaler()
    X_train = df_train[Q1_COLS].values.astype(float)
    X_train_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    pca.fit(X_train_scaled)
    components = pca.components_
    var_ratio = pca.explained_variance_ratio_

    def compute_H(df):
        X = df[Q1_COLS].values.astype(float)
        scores = X @ components.T
        H = scores @ var_ratio
        return H

    def q1_predict_ratio(df):
        H = compute_H(df)
        H = np.maximum(H, 1e-10)
        ratio = H / H.sum()
        return ratio, H

    return q1_predict_ratio, compute_H, scaler, pca, components, var_ratio


def q2_exp_smooth_ratio(hist_ratios, target_offset=1):
    n = len(hist_ratios)
    if n == 0: return np.nan
    if n == 1: return hist_ratios[0]
    try:
        model = SimpleExpSmoothing(hist_ratios)
        fit = model.fit(optimized=True)
        pred = fit.forecast(target_offset)
        return float(pred[-1] if target_offset > 1 else pred[0])
    except Exception:
        if n >= 2:
            slope = (hist_ratios[-1] - hist_ratios[0]) / (n - 1)
            return float(hist_ratios[-1] + slope * target_offset)
        return hist_ratios[-1]


def q2_moving_average_ratio(hist_ratios, window=3):
    """移动平均法预测比例"""
    n = len(hist_ratios)
    if n == 0: return np.nan
    if n == 1: return hist_ratios[0]
    w = min(window, n)
    return float(np.mean(hist_ratios[-w:]))


def budget_adjust(total_pred, B_budget):
    total_pred = float(total_pred)
    if total_pred <= B_budget + 1e-9:
        return total_pred, total_pred * RATIOS
    return B_budget, B_budget * RATIOS


def evaluate_prediction(pred, actual, name):
    n = len(actual)
    residual = pred - actual
    print(f"\n{'='*60}")
    print(f">>> {name} 评估结果")
    print(f"{'='*60}")
    t_stat, p_mean = stats.ttest_rel(pred, actual)
    mean_conclusion = "通过" if p_mean > ALPHA else "未通过"
    print(f"配对t检验: p={p_mean:.4f} → {mean_conclusion}")
    var_pred = np.var(pred, ddof=1)
    var_actual = np.var(actual, ddof=1)
    if var_pred > var_actual:
        f_stat = var_pred / var_actual
        p_tail = 1 - stats.f.cdf(f_stat, n-1, n-1)
    else:
        f_stat = var_actual / var_pred
        p_tail = 1 - stats.f.cdf(f_stat, n-1, n-1)
    p_var_f = 2 * min(p_tail, 1-p_tail)
    p_var_f = min(p_var_f, 1.0)
    f_conclusion = "通过" if p_var_f > ALPHA else "未通过"
    print(f"F检验: p={p_var_f:.4f} → {f_conclusion}")
    stat_lev, p_lev = stats.levene(pred, actual)
    lev_conclusion = "通过" if p_lev > ALPHA else "未通过"
    print(f"Levene检验: p={p_lev:.4f} → {lev_conclusion}")
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return {'p_mean': p_mean, 'mean_conclusion': mean_conclusion,
            'p_var_f': p_var_f, 'f_conclusion': f_conclusion,
            'p_lev': p_lev, 'lev_conclusion': lev_conclusion,
            'mae': mae, 'rmse': rmse, 'r2': r2}


def load_ols_fitted():
    """读取官方OLS拟合结果（不含组织员）"""
    xlsx_path = '测算数据（两个参数）.xlsx'
    df_ols = pd.read_excel(xlsx_path, sheet_name='不含组织员', header=0)
    def normalize_name(name):
        if pd.isna(name): return ''
        return str(name).replace(' ', '').replace('，', ',').replace(',', ',')
    df_ols['学院_标准化'] = df_ols['学院'].apply(normalize_name)
    return df_ols


def compare_with_ols(pred_total, dept_names, ols_df, strategy_name):
    """将预测结果与OLS拟合结果进行比较"""
    def normalize_name(name):
        return str(name).replace(' ', '').replace('，', ',').replace(',', ',')

    pred_map = {normalize_name(d): pred_total[i] for i, d in enumerate(dept_names)}

    comparisons = []
    for _, row in ols_df.iterrows():
        std_name = row['学院_标准化']
        if std_name in pred_map:
            pred_val = pred_map[std_name]
            fitted_val = row['拟合数']
            diff = pred_val - fitted_val
            comparisons.append({
                '学院': row['学院'],
                '管理岗现状': row['管理岗现状'],
                'OLS拟合数': fitted_val,
                '预测值': round(pred_val, 4),
                '差异': round(diff, 4),
                '差异百分比': round(diff / fitted_val * 100, 2) if fitted_val != 0 else np.nan
            })

    comp_df = pd.DataFrame(comparisons)
    print(f"\n{'='*60}")
    print(f">>> {strategy_name} 与官方OLS拟合结果对比")
    print(f"{'='*60}")
    print(comp_df.to_string(index=False))

    ols_vals = comp_df['OLS拟合数'].values
    pred_vals = comp_df['预测值'].values
    mae_ols = np.mean(np.abs(pred_vals - ols_vals))
    rmse_ols = np.sqrt(np.mean((pred_vals - ols_vals)**2))
    print(f"\n相对OLS的MAE={mae_ols:.4f}, RMSE={rmse_ols:.4f}")

    return comp_df


# ==================== 主程序 ====================
if __name__ == '__main__':
    dfs, dept_names, dept_cats, available_years = load_and_preprocess()
    n_depts = len(dept_names)
    print(f"学院数量: {n_depts}")
    print(f"可用年份: {available_years}")
    print(f"排除学院: {EXCLUDE_NAMES}")

    actual_ratios, actual_totals = compute_actual_ratios(dfs)
    for year in sorted(dfs.keys()):
        print(f"{year}年总编制: {actual_totals[year]}, 比例和: {actual_ratios[year].sum():.6f}")

    B_total_2026 = actual_totals[2026]
    budget_2026 = dfs[2026]['预算约束人数'].values.astype(float)
    actual_2026 = dfs[2026]['编制人数'].values.astype(float)

    # Q1模型：以2025年数据为训练集构建PCA
    q1_predict_ratio, compute_H, scaler, pca, components, var_ratio = build_q1_model(dfs[2025])
    print(f"\nPCA方差贡献率: {var_ratio}")
    print(f"PCA累计贡献率: {var_ratio.sum():.4f}")

    # 加载OLS拟合数据
    ols_df = load_ols_fitted()

    # 历史年份（用于Q2）：所有小于2026且存在的年份
    hist_years = sorted([y for y in available_years if y < 2026])
    print(f"Q2历史年份: {hist_years}")

    # ==================== 两种Q2策略，仅统一权重 ====================
    q2_methods = {
        'exp': ('简单指数平滑', q2_exp_smooth_ratio),
        'ma': ('移动平均法', q2_moving_average_ratio)
    }

    all_results = []

    for q2_key, (q2_name, q2_func) in q2_methods.items():
        print(f"\n{'#'*70}")
        print(f"# Q2策略: {q2_name}")
        print(f"{'#'*70}")

        # Q1预测2026年比例（PCA用2025年fit，输入2026年特征）
        ratio_q1_pred, H_pred = q1_predict_ratio(dfs[2026])

        # Q2预测2026年比例（基于历史实际比例）
        ratio_q2_pred = np.zeros(n_depts)
        for i in range(n_depts):
            hist = [actual_ratios[y][i] for y in hist_years]
            ratio_q2_pred[i] = q2_func(hist, 1)
        ratio_q2_pred = np.maximum(ratio_q2_pred, 0)
        if ratio_q2_pred.sum() > 0:
            ratio_q2_pred = ratio_q2_pred / ratio_q2_pred.sum()
        else:
            ratio_q2_pred = np.ones(n_depts) / n_depts

        # 实际2026年比例
        ratio_actual = actual_ratios[2026]

        # 计算各学院差值
        diff_q1 = ratio_q1_pred - ratio_actual
        diff_q2 = ratio_q2_pred - ratio_actual

        print(f"\n--- 2025年训练 → 2026年验证 ---")
        print(f"Q1比例MAE={np.mean(np.abs(diff_q1)):.6f}")
        print(f"Q2比例MAE={np.mean(np.abs(diff_q2)):.6f}")

        # 基于各学院差值计算总方差，确定统一权重
        var_q1_total = np.mean(diff_q1 ** 2)
        var_q2_total = np.mean(diff_q2 ** 2)
        var_q1_total = max(var_q1_total, 1e-10)
        var_q2_total = max(var_q2_total, 1e-10)
        w_q1_global = (1 / var_q1_total) / (1 / var_q1_total + 1 / var_q2_total)
        w_q2_global = 1 - w_q1_global

        print(f"\n{'='*60}")
        print(f"组合定权结果 ({q2_name}) — 仅统一权重")
        print(f"{'='*60}")
        print(f"统一权重: w_Q1={w_q1_global:.4f}, w_Q2={w_q2_global:.4f}")

        # 2026年最终组合预测（统一权重）
        ratio_comb_global = w_q1_global * ratio_q1_pred + w_q2_global* ratio_q2_pred
        ratio_comb_global = ratio_comb_global / ratio_comb_global.sum()
        pred_global = ratio_comb_global * B_total_2026

        # 预算约束
        pred_global_adj_total = np.zeros(n_depts)
        pred_global_adj_split = np.zeros((n_depts, 3))
        for i in range(n_depts):
            adj_total, split = budget_adjust(pred_global[i], budget_2026[i])
            pred_global_adj_total[i] = adj_total
            pred_global_adj_split[i] = split

        # 评估（与2026年真实数据比较）
        metrics_global = evaluate_prediction(pred_global, actual_2026, f"统一权重-{q2_name}")

        # 与OLS拟合结果比较
        comp_global = compare_with_ols(pred_global, dept_names, ols_df, f"统一权重-{q2_name}")

        # 保存结果
        result_global = pd.DataFrame({
            '大类': dept_cats, '部门名称': dept_names,
            '预算约束人数': budget_2026.astype(int), '实际编制': actual_2026.astype(int),
            'Q1比例': np.round(ratio_q1_pred, 6), 'Q2比例': np.round(ratio_q2_pred, 6),
            '组合比例': np.round(ratio_comb_global, 6),
            '约束前_总编制': np.round(pred_global, 2),
            '约束前_正科': np.round(pred_global * 0.2, 2),
            '约束前_副科': np.round(pred_global * 0.3, 2),
            '约束前_科级以下': np.round(pred_global * 0.5, 2),
            '约束后_总编制': np.round(pred_global_adj_total, 2),
            '约束后_正科': np.round(pred_global_adj_split[:, 0], 2),
            '约束后_副科': np.round(pred_global_adj_split[:, 1], 2),
            '约束后_科级以下': np.round(pred_global_adj_split[:, 2], 2),
        })
        test_summary = pd.DataFrame({
            '检验/指标': ['配对t检验(p值)', 'F检验(p值)', 'Levene检验(p值)', 'MAE', 'RMSE', 'R²'],
            '统一权重': [
                f"{metrics_global['p_mean']:.4f} ({metrics_global['mean_conclusion']})",
                f"{metrics_global['p_var_f']:.4f} ({metrics_global['f_conclusion']})",
                f"{metrics_global['p_lev']:.4f} ({metrics_global['lev_conclusion']})",
                f"{metrics_global['mae']:.4f}", f"{metrics_global['rmse']:.4f}", f"{metrics_global['r2']:.4f}",
            ]
        })

        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
        os.makedirs(script_dir, exist_ok=True)

        suffix = f"_{q2_key}"
        result_global.to_csv(os.path.join(script_dir, f'result_strategy{suffix}.csv'), index=False, encoding='utf-8-sig')
        test_summary.to_csv(os.path.join(script_dir, f'result_metrics{suffix}.csv'), index=False, encoding='utf-8-sig')
        comp_global.to_csv(os.path.join(script_dir, f'result_ols_comp{suffix}.csv'), index=False, encoding='utf-8-sig')

        print(f"\n[完成] {q2_name} 结果已保存")
        print(f"  - result_strategy{suffix}.csv")
        print(f"  - result_metrics{suffix}.csv")
        print(f"  - result_ols_comp{suffix}.csv")

        all_results.append({
            'strategy': q2_name,
            'metrics': metrics_global,
            'ols_mae': np.mean(np.abs(comp_global['预测值'].values - comp_global['OLS拟合数'].values)),
            'ols_rmse': np.sqrt(np.mean((comp_global['预测值'].values - comp_global['OLS拟合数'].values)**2))
        })

    # 汇总对比
    print(f"\n{'='*70}")
    print("# 两种Q2策略汇总对比")
    print(f"{'='*70}")
    for r in all_results:
        print(f"\n{r['strategy']}:")
        print(f"  相对真实数据: MAE={r['metrics']['mae']:.4f}, RMSE={r['metrics']['rmse']:.4f}, R²={r['metrics']['r2']:.4f}")
        print(f"  相对OLS拟合: MAE={r['ols_mae']:.4f}, RMSE={r['ols_rmse']:.4f}")
