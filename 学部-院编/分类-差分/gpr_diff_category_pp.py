import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from itertools import combinations

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_2024_PATH = os.path.join(BASE_DIR, 'data2024.csv')
TRAIN_2025_PATH = os.path.join(BASE_DIR, 'data2025.csv')
TRAIN_2026_PATH = os.path.join(BASE_DIR, 'data2026.csv')
OFFICIAL_OLS_XLSX = os.path.join(BASE_DIR, '测算数据（两个参数）_backup.xlsx')

# 差分不可用的特征列（data2024中缺失）：留学生人数、长期留学生人数、折合学生数、预算约束人数、劳动合同制专职科研人员、一年制科研助理
# 核函数配置
KERNEL = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
    length_scale=1.0, length_scale_bounds=(1e-2, 10.0)
) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

RATIOS = np.array([0.2, 0.3, 0.5])
ALPHA = 0.05

# ==================== 中文字体设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 工具函数 ====================
def load_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    return df.reset_index(drop=True)


def identify_zero_diff_depts(X_diff):
    """识别差分特征全为0的学院"""
    mask = (X_diff == 0).all(axis=1)
    return mask


def match_official_ols(dept_names, df_official):
    """匹配官方OLS拟合值（参数1和参数2），支持模糊匹配"""
    official_param1 = []
    official_param2 = []
    for name in dept_names:
        match = df_official[df_official['单位'] == name]
        if len(match) > 0:
            official_param1.append(match['拟合数\n（参数1）'].values[0])
            official_param2.append(match['拟合数\n（参数2）'].values[0])
            continue
        for idx, off_name in enumerate(df_official['单位']):
            if pd.isna(off_name):
                continue
            if name in off_name or off_name in name:
                official_param1.append(df_official.iloc[idx]['拟合数\n（参数1）'])
                official_param2.append(df_official.iloc[idx]['拟合数\n（参数2）'])
                break
        else:
            official_param1.append(np.nan)
            official_param2.append(np.nan)
    return np.array(official_param1), np.array(official_param2)


def split_levels(pred_total, org_count):
    non_org = max(0, pred_total - org_count)
    split = non_org * RATIOS
    return split


def evaluate_metrics(y_pred, y_actual):
    n = len(y_actual)
    residual = y_pred - y_actual
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))
    valid_mask = (y_actual != 0) & np.isfinite(y_actual) & np.isfinite(residual)
    if np.sum(valid_mask) > 0:
        mape = np.mean(np.abs(residual[valid_mask] / y_actual[valid_mask])) * 100
    else:
        mape = np.nan
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    t_stat, p_mean = stats.ttest_rel(y_pred, y_actual)
    mean_conclusion = "通过" if p_mean > ALPHA else "未通过"

    var_pred = np.var(y_pred, ddof=1)
    var_actual = np.var(y_actual, ddof=1)
    if var_pred > var_actual:
        f_stat = var_pred / var_actual
        p_tail = 1 - stats.f.cdf(f_stat, n - 1, n - 1)
    else:
        f_stat = var_actual / var_pred
        p_tail = 1 - stats.f.cdf(f_stat, n - 1, n - 1)
    p_var_f = 2 * min(p_tail, 1 - p_tail)
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


def calc_mae_rmse_vs_official(y_pred, official_vals):
    valid_mask = ~np.isnan(official_vals)
    if np.sum(valid_mask) == 0:
        return np.nan, np.nan
    diff = y_pred[valid_mask] - official_vals[valid_mask]
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    return mae, rmse


def extract_gpr_params(gpr):
    try:
        sig_var = gpr.kernel_.k1.k1.constant_value
        length_scale = gpr.kernel_.k1.k2.length_scale
        noise_var = gpr.kernel_.k2.noise_level
        return {
            'signal_variance': sig_var,
            'length_scale': length_scale,
            'noise_variance': noise_var,
            'log_marginal_likelihood': gpr.log_marginal_likelihood_value_,
            'alpha': gpr.alpha
        }
    except Exception:
        return {
            'signal_variance': np.nan,
            'length_scale': np.nan,
            'noise_variance': np.nan,
            'log_marginal_likelihood': np.nan,
            'alpha': gpr.alpha
        }


# ==================== 部分池化预测（差分版）====================
def partial_pooling_predict_diff(df_base, df_target, gpr_idx, zero_idx, feature_cols, dept_names, n_depts):
    """
    按大类进行部分池化预测（差分版本）
    返回：全局预测差分、部分池化预测差分、全局超参数、各组超参数、收缩权重、scaler
    """
    X_diff = df_target[feature_cols].values.astype(float) - df_base[feature_cols].values.astype(float)
    y_diff = df_target['编制人数'].values.astype(float) - df_base['编制人数'].values.astype(float)

    X_train = X_diff[gpr_idx]
    y_train = y_diff[gpr_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    gpr_global = GaussianProcessRegressor(
        kernel=KERNEL, n_restarts_optimizer=10, normalize_y=True,
        alpha=1e-2, random_state=RANDOM_STATE
    )
    gpr_global.fit(X_train_scaled, y_train)
    global_params = extract_gpr_params(gpr_global)

    y_diff_global = np.zeros(n_depts)
    y_diff_global[zero_idx] = y_diff[zero_idx]
    y_pred_global, _ = gpr_global.predict(X_train_scaled, return_std=True)
    y_diff_global[gpr_idx] = y_pred_global

    categories = df_target.loc[gpr_idx, '大类'].values
    unique_cats = np.unique(categories)
    K = len(unique_cats)
    n_global = len(gpr_idx)
    n_bar = n_global / K

    group_params = {}
    group_preds = {}
    lambdas = {}

    for cat in unique_cats:
        cat_mask = categories == cat
        cat_idx_in_gpr = np.where(cat_mask)[0]
        cat_global_idx = [gpr_idx[i] for i in cat_idx_in_gpr]
        n_k = len(cat_idx_in_gpr)

        if n_k < 2:
            lambdas[cat] = 0.0
            group_preds[cat] = y_diff_global.copy()
            group_params[cat] = global_params.copy()
            continue

        X_train_cat = X_diff[cat_global_idx]
        y_train_cat = y_diff[cat_global_idx]
        X_train_cat_scaled = scaler.transform(X_train_cat)

        gpr_cat = GaussianProcessRegressor(
            kernel=KERNEL, n_restarts_optimizer=10, normalize_y=True,
            alpha=1e-2, random_state=RANDOM_STATE
        )
        gpr_cat.fit(X_train_cat_scaled, y_train_cat)
        group_params[cat] = extract_gpr_params(gpr_cat)

        y_pred_cat, _ = gpr_cat.predict(X_train_cat_scaled, return_std=True)
        y_cat_full = np.zeros(n_depts)
        y_cat_full[zero_idx] = y_diff[zero_idx]
        for j, idx in enumerate(cat_global_idx):
            y_cat_full[idx] = y_pred_cat[j]
        group_preds[cat] = y_cat_full

        lambdas[cat] = n_k / (n_k + n_bar)

    y_diff_partial = np.zeros(n_depts)
    y_diff_partial[zero_idx] = y_diff[zero_idx]

    for i in gpr_idx:
        cat = df_target.loc[i, '大类']
        lam = lambdas.get(cat, 0.0)
        y_diff_partial[i] = lam * group_preds[cat][i] + (1 - lam) * y_diff_global[i]

    return y_diff_global, y_diff_partial, global_params, group_params, lambdas, scaler


def generate_feature_combos():
    """
    差分版本特征组合（已排除data2024缺失特征）：
    - 去掉：折合学生数、留学生人数、长期留学生人数、预算约束人数、劳动合同制专职科研人员、一年制科研助理
    - 保留：本+研+师 及其公共课扩展、在编扩展、博士后、派遣、高研院、柔性
    - 不考虑：双轨制
    """
    bases = [
        {'name': '本+研+师', 'cols': ['本科生', '研究生', '在编-教师']},
    ]
    public_options = [
        {'name': '', 'cols': []},
        {'name': '+公课数量', 'cols': ['公共课数量']},
        {'name': '+是否公课', 'cols': ['是否公共课']},
    ]
    new_groups = [
        {'name': '+在编', 'cols': ['在编-教辅', '在编-思政', '在编-管理', '在编-工勤']},
        {'name': '+博士后', 'cols': ['博士后']},
        {'name': '+派遣', 'cols': ['学校经费派遣', '部门经费派遣', '项目经费派遣']},
        {'name': '+高研院', 'cols': ['可上报高研院', '不可上报高研院']},
        {'name': '+柔性', 'cols': ['可上报柔性', '不可上报柔性']},
    ]

    combos = []
    for base in bases:
        for pub in public_options:
            for r in range(len(new_groups) + 1):
                for subset in combinations(new_groups, r):
                    name = base['name'] + pub['name']
                    cols = base['cols'] + pub['cols']
                    for g in subset:
                        name += g['name']
                        cols += g['cols']
                    combos.append({'name': name, 'cols': cols})
    return combos


# ==================== 主程序 ====================
def main():
    COMBOS = generate_feature_combos()
    print(f"[信息] 共生成 {len(COMBOS)} 种特征组合（差分版本，已排除data2024缺失特征）")

    df_official = pd.read_excel(OFFICIAL_OLS_XLSX, sheet_name='含组织员')
    df_2024 = load_data(TRAIN_2024_PATH)
    df_2025 = load_data(TRAIN_2025_PATH)
    df_2026 = load_data(TRAIN_2026_PATH)

    dept_names = df_2025['部门名称'].tolist()
    df_2024 = df_2024.set_index('部门名称').reindex(dept_names).reset_index()
    df_2026 = df_2026.set_index('部门名称').reindex(dept_names).reset_index()

    official_param1, official_param2 = match_official_ols(dept_names, df_official)
    n_depts = len(dept_names)

    org_counts_25 = df_2025['组织员人数'].values.astype(int)
    org_counts_26 = df_2026['组织员人数'].values.astype(int)

    y_actual_25 = df_2025['编制人数'].values.astype(float)
    y_actual_26 = df_2026['编制人数'].values.astype(float)

    base_bianzhi_24 = df_2024['编制人数'].values.astype(float)
    base_bianzhi_25 = df_2025['编制人数'].values.astype(float)

    pred_24to25 = {}
    metrics_24to25 = {}

    pred_25to26_global = {}
    pred_25to26_partial = {}
    std_25to26 = {}
    metrics_25to26_global = {}
    metrics_25to26_partial = {}
    pp_results = {}

    # ==================== 模式1：2024→2025 差分验证（仅全局）====================
    print("\n" + "=" * 80)
    print(f">>> 模式一：2024→2025 差分训练 → 2025年验证（共{len(COMBOS)}种组合，仅全局GPR）")
    print("=" * 80)

    for combo_idx, combo in enumerate(COMBOS):
        combo_name = combo['name']
        feature_cols = combo['cols']

        missing_cols = [c for c in feature_cols if c not in df_2024.columns or c not in df_2025.columns]
        if missing_cols:
            continue

        X_diff = df_2025[feature_cols].values.astype(float) - df_2024[feature_cols].values.astype(float)
        y_diff = y_actual_25 - base_bianzhi_24

        zero_mask = identify_zero_diff_depts(X_diff)
        gpr_idx = np.where(~zero_mask)[0].tolist()
        zero_idx = np.where(zero_mask)[0].tolist()

        y_pred_diff = np.zeros(n_depts)
        y_pred_diff[zero_idx] = y_diff[zero_idx]

        if len(gpr_idx) > 0:
            X_train = X_diff[gpr_idx]
            y_train = y_diff[gpr_idx]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            gpr = GaussianProcessRegressor(
                kernel=KERNEL, n_restarts_optimizer=10, normalize_y=True,
                alpha=1e-2, random_state=RANDOM_STATE
            )
            gpr.fit(X_train_scaled, y_train)
            y_pred_gpr, _ = gpr.predict(X_train_scaled, return_std=True)
            y_pred_diff[gpr_idx] = y_pred_gpr

        y_pred_total = base_bianzhi_24 + y_pred_diff
        y_pred_total = np.maximum(y_pred_total, 0)
        pred_24to25[combo_name] = y_pred_total.copy()

        metrics_actual = evaluate_metrics(y_pred_total, y_actual_25)
        metrics_24to25[combo_name] = {
            'mae_actual': metrics_actual['mae'],
            'rmse_actual': metrics_actual['rmse'],
            'r2': metrics_actual['r2'],
            'mape': metrics_actual['mape'],
            't_p': metrics_actual['p_mean'],
            'mean_conclusion': metrics_actual['mean_conclusion'],
            'p_var_f': metrics_actual['p_var_f'],
            'f_conclusion': metrics_actual['f_conclusion'],
            'p_lev': metrics_actual['p_lev'],
            'lev_conclusion': metrics_actual['lev_conclusion'],
            'mae_off1': np.nan,
            'rmse_off1': np.nan,
        }

        if (combo_idx + 1) % 50 == 0 or combo_idx == len(COMBOS) - 1:
            print(f"  进度: {combo_idx + 1}/{len(COMBOS)}")

    # ==================== 模式2：2025→2026 差分自预测（全局+部分池化）====================
    print("\n" + "=" * 80)
    print(f">>> 模式二：2025→2026 差分训练 → 2026年自预测（全局+部分池化，共{len(COMBOS)}种组合）")
    print("=" * 80)

    for combo_idx, combo in enumerate(COMBOS):
        combo_name = combo['name']
        feature_cols = combo['cols']

        missing_cols = [c for c in feature_cols if c not in df_2025.columns or c not in df_2026.columns]
        if missing_cols:
            continue

        X_diff = df_2026[feature_cols].values.astype(float) - df_2025[feature_cols].values.astype(float)
        y_diff = y_actual_26 - base_bianzhi_25

        zero_mask = identify_zero_diff_depts(X_diff)
        gpr_idx = np.where(~zero_mask)[0].tolist()
        zero_idx = np.where(zero_mask)[0].tolist()

        # 全局GPR
        y_pred_diff_global = np.zeros(n_depts)
        y_std_diff = np.zeros(n_depts)
        y_pred_diff_global[zero_idx] = y_diff[zero_idx]

        if len(gpr_idx) > 0:
            X_train = X_diff[gpr_idx]
            y_train = y_diff[gpr_idx]
            X_pred = X_diff[gpr_idx]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_pred_scaled = scaler.transform(X_pred)
            gpr = GaussianProcessRegressor(
                kernel=KERNEL, n_restarts_optimizer=10, normalize_y=True,
                alpha=1e-2, random_state=RANDOM_STATE
            )
            gpr.fit(X_train_scaled, y_train)
            y_pred_gpr, y_std_gpr = gpr.predict(X_pred_scaled, return_std=True)
            y_pred_diff_global[gpr_idx] = y_pred_gpr
            y_std_diff[gpr_idx] = y_std_gpr

        y_pred_global_total = base_bianzhi_25 + y_pred_diff_global
        y_pred_global_total = np.maximum(y_pred_global_total, 0)
        pred_25to26_global[combo_name] = y_pred_global_total.copy()
        std_25to26[combo_name] = y_std_diff.copy()

        metrics_global_actual = evaluate_metrics(y_pred_global_total, y_actual_26)
        mae_off1_g, rmse_off1_g = calc_mae_rmse_vs_official(y_pred_global_total, official_param1)
        mae_off2_g, rmse_off2_g = calc_mae_rmse_vs_official(y_pred_global_total, official_param2)

        metrics_25to26_global[combo_name] = {
            'mae_actual': metrics_global_actual['mae'],
            'rmse_actual': metrics_global_actual['rmse'],
            'r2': metrics_global_actual['r2'],
            'mape': metrics_global_actual['mape'],
            't_p': metrics_global_actual['p_mean'],
            'mean_conclusion': metrics_global_actual['mean_conclusion'],
            'p_var_f': metrics_global_actual['p_var_f'],
            'f_conclusion': metrics_global_actual['f_conclusion'],
            'p_lev': metrics_global_actual['p_lev'],
            'lev_conclusion': metrics_global_actual['lev_conclusion'],
            'mae_off1': mae_off1_g,
            'rmse_off1': rmse_off1_g,
            'mae_off2': mae_off2_g,
            'rmse_off2': rmse_off2_g,
        }

        # 部分池化
        y_diff_global, y_diff_partial, global_params, group_params, lambdas, _ =             partial_pooling_predict_diff(df_2025, df_2026, gpr_idx, zero_idx, feature_cols, dept_names, n_depts)

        y_pred_partial_total = base_bianzhi_25 + y_diff_partial
        y_pred_partial_total = np.maximum(y_pred_partial_total, 0)
        pred_25to26_partial[combo_name] = y_pred_partial_total.copy()

        pp_results[combo_name] = {
            'y_diff_global': y_diff_global,
            'y_diff_partial': y_diff_partial,
            'global_params': global_params,
            'group_params': group_params,
            'lambdas': lambdas
        }

        metrics_pp_actual = evaluate_metrics(y_pred_partial_total, y_actual_26)
        mae_off1_pp, rmse_off1_pp = calc_mae_rmse_vs_official(y_pred_partial_total, official_param1)
        mae_off2_pp, rmse_off2_pp = calc_mae_rmse_vs_official(y_pred_partial_total, official_param2)

        metrics_25to26_partial[combo_name] = {
            'mae_actual': metrics_pp_actual['mae'],
            'rmse_actual': metrics_pp_actual['rmse'],
            'r2': metrics_pp_actual['r2'],
            'mape': metrics_pp_actual['mape'],
            't_p': metrics_pp_actual['p_mean'],
            'mean_conclusion': metrics_pp_actual['mean_conclusion'],
            'p_var_f': metrics_pp_actual['p_var_f'],
            'f_conclusion': metrics_pp_actual['f_conclusion'],
            'p_lev': metrics_pp_actual['p_lev'],
            'lev_conclusion': metrics_pp_actual['lev_conclusion'],
            'mae_off1': mae_off1_pp,
            'rmse_off1': rmse_off1_pp,
            'mae_off2': mae_off2_pp,
            'rmse_off2': rmse_off2_pp,
        }

        if (combo_idx + 1) % 50 == 0 or combo_idx == len(COMBOS) - 1:
            print(f"  进度: {combo_idx + 1}/{len(COMBOS)}")

    # ==================== 找出最优组合 ====================
    sorted_mode1 = sorted(metrics_24to25.items(), key=lambda x: x[1]['mae_actual'])
    best3_mode1 = sorted_mode1[:3]

    sorted_mode2_global = sorted(metrics_25to26_global.items(), key=lambda x: x[1]['mae_off1'])
    best3_mode2_global = sorted_mode2_global[:3]

    sorted_mode2_pp = sorted(metrics_25to26_partial.items(), key=lambda x: x[1]['mae_off1'])
    best3_mode2_pp = sorted_mode2_pp[:3]

    print("\n" + "=" * 80)
    print(">>> 模式一最优3个组合（按MAE_vs实际排序）")
    print("=" * 80)
    for i, (name, m) in enumerate(best3_mode1):
        print(f"\n  第{i+1}名: {name}")
        print(f"    vs实际: MAE={m['mae_actual']:.3f}, RMSE={m['rmse_actual']:.3f}, R²={m['r2']:.3f}")

    print("\n" + "=" * 80)
    print(">>> 模式二全局GPR最优3个组合（按MAE_vs backup参数1排序）")
    print("=" * 80)
    for i, (name, m) in enumerate(best3_mode2_global):
        print(f"\n  第{i+1}名: {name}")
        print(f"    MAE_vs参数1={m['mae_off1']:.3f}, RMSE_vs参数1={m['rmse_off1']:.3f}")
        print(f"    MAE_vs参数2={m['mae_off2']:.3f}, RMSE_vs参数2={m['rmse_off2']:.3f}")
        print(f"    vs实际: MAE={m['mae_actual']:.3f}, RMSE={m['rmse_actual']:.3f}, R²={m['r2']:.3f}")

    print("\n" + "=" * 80)
    print(">>> 模式二部分池化最优3个组合（按MAE_vs backup参数1排序）")
    print("=" * 80)
    for i, (name, m) in enumerate(best3_mode2_pp):
        print(f"\n  第{i+1}名: {name}")
        print(f"    MAE_vs参数1={m['mae_off1']:.3f}, RMSE_vs参数1={m['rmse_off1']:.3f}")
        print(f"    MAE_vs参数2={m['mae_off2']:.3f}, RMSE_vs参数2={m['rmse_off2']:.3f}")
        print(f"    vs实际: MAE={m['mae_actual']:.3f}, RMSE={m['rmse_actual']:.3f}, R²={m['r2']:.3f}")
        print(f"    均值检验: t_p={m['t_p']:.3f} ({m['mean_conclusion']})")
        print(f"    方差检验: F_p={m['p_var_f']:.3f} ({m['f_conclusion']})")

    # ==================== 输出模式2最优3个组合的detail（含部分池化）====================
    print("\n[保存] 模式二最优3个组合的detail CSV（含部分池化）...")
    for i, (combo_name, _) in enumerate(best3_mode2_pp):
        pred_global = pred_25to26_global[combo_name]
        pred_pp = pred_25to26_partial[combo_name]
        std = std_25to26[combo_name]

        split_global = np.zeros((n_depts, 3))
        split_pp = np.zeros((n_depts, 3))
        for j in range(n_depts):
            split_global[j] = split_levels(pred_global[j], org_counts_26[j])
            split_pp[j] = split_levels(pred_pp[j], org_counts_26[j])

        detail_df = pd.DataFrame({
            '部门名称': dept_names,
            '大类': df_2026['大类'].values,
            '2026实际编制': y_actual_26.astype(int),
            '全局GPR预测': np.round(pred_global, 2),
            '部分池化预测': np.round(pred_pp, 2),
            'GPR_std': np.round(std, 2),
            '官方参数1_backup': ['null' if pd.isna(v) else round(v, 2) for v in official_param1],
            '官方参数2_backup': ['null' if pd.isna(v) else round(v, 2) for v in official_param2],
            '正科_全局(20%)': np.round(split_global[:, 0], 2),
            '副科_全局(30%)': np.round(split_global[:, 1], 2),
            '科级以下_全局(50%)': np.round(split_global[:, 2], 2),
            '正科_部分池化(20%)': np.round(split_pp[:, 0], 2),
            '副科_部分池化(30%)': np.round(split_pp[:, 1], 2),
            '科级以下_部分池化(50%)': np.round(split_pp[:, 2], 2),
        })
        safe_name = combo_name.replace('+', '_')
        fname = f'gpr_diff_category_pp_26self_{safe_name}.csv'
        detail_df.to_csv(os.path.join(BASE_DIR, fname), index=False, encoding='utf-8-sig')
        print(f"  [保存] {fname}")

    # ==================== 生成模式2汇总表（含部分池化与超参数）====================
    print("\n[保存] 模式二汇总表（含部分池化与超参数）...")
    op1_str = ['null' if pd.isna(v) else str(round(v, 2)) for v in official_param1]
    op2_str = ['null' if pd.isna(v) else str(round(v, 2)) for v in official_param2]

    summary_data = {
        '部门名称': dept_names,
        '大类': df_2026['大类'].values,
        '官方参数1_backup': op1_str,
        '官方参数2_backup': op2_str,
        '2026实际编制': y_actual_26.astype(int),
    }

    for i, (combo_name, _) in enumerate(best3_mode2_pp):
        safe_name = combo_name.replace('+', '_')
        pred_g = pred_25to26_global[combo_name]
        pred_pp = pred_25to26_partial[combo_name]
        diff1_g = pred_g - official_param1
        diff2_g = pred_g - official_param2
        diff1_pp = pred_pp - official_param1
        diff2_pp = pred_pp - official_param2

        summary_data[f'{safe_name}_全局预测'] = np.round(pred_g, 2)
        summary_data[f'{safe_name}_全局vs参数1差值'] = np.round(diff1_g, 2)
        summary_data[f'{safe_name}_全局vs参数2差值'] = np.round(diff2_g, 2)
        summary_data[f'{safe_name}_部分池化预测'] = np.round(pred_pp, 2)
        summary_data[f'{safe_name}_部分池化vs参数1差值'] = np.round(diff1_pp, 2)
        summary_data[f'{safe_name}_部分池化vs参数2差值'] = np.round(diff2_pp, 2)

    summary_df = pd.DataFrame(summary_data)

    metrics_map_global = {
        '【汇总】全局_R²': 'r2',
        '【汇总】全局_MAPE(%)': 'mape',
        '【汇总】全局_t_p(均值检验)': 't_p',
        '【汇总】全局_均值检验结论': 'mean_conclusion',
        '【汇总】全局_F_p(方差检验)': 'p_var_f',
        '【汇总】全局_方差检验结论': 'f_conclusion',
        '【汇总】全局_Levene_p': 'p_lev',
        '【汇总】全局_Levene检验结论': 'lev_conclusion',
        '【汇总】全局_MAE_vs参数1': 'mae_off1',
        '【汇总】全局_RMSE_vs参数1': 'rmse_off1',
        '【汇总】全局_MAE_vs参数2': 'mae_off2',
        '【汇总】全局_RMSE_vs参数2': 'rmse_off2',
    }

    metric_rows = []
    for row_name, metric_key in metrics_map_global.items():
        row = {'部门名称': row_name, '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2_pp):
            safe_name = combo_name.replace('+', '_')
            val = metrics_25to26_global[combo_name].get(metric_key, '')
            if isinstance(val, float):
                val = round(val, 3)
            for suffix in [f'{safe_name}_全局预测', f'{safe_name}_全局vs参数1差值', 
                           f'{safe_name}_全局vs参数2差值', f'{safe_name}_部分池化预测',
                           f'{safe_name}_部分池化vs参数1差值', f'{safe_name}_部分池化vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    pp_metrics_map = {
        '【汇总】部分池化_R²': 'r2',
        '【汇总】部分池化_MAPE(%)': 'mape',
        '【汇总】部分池化_t_p(均值检验)': 't_p',
        '【汇总】部分池化_均值检验结论': 'mean_conclusion',
        '【汇总】部分池化_F_p(方差检验)': 'p_var_f',
        '【汇总】部分池化_方差检验结论': 'f_conclusion',
        '【汇总】部分池化_Levene_p': 'p_lev',
        '【汇总】部分池化_Levene检验结论': 'lev_conclusion',
        '【汇总】部分池化_MAE_vs参数1': 'mae_off1',
        '【汇总】部分池化_RMSE_vs参数1': 'rmse_off1',
        '【汇总】部分池化_MAE_vs参数2': 'mae_off2',
        '【汇总】部分池化_RMSE_vs参数2': 'rmse_off2',
    }
    for row_name, metric_key in pp_metrics_map.items():
        row = {'部门名称': row_name, '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2_pp):
            safe_name = combo_name.replace('+', '_')
            val = metrics_25to26_partial[combo_name].get(metric_key, '')
            if isinstance(val, float):
                val = round(val, 3)
            for suffix in [f'{safe_name}_全局预测', f'{safe_name}_全局vs参数1差值', 
                           f'{safe_name}_全局vs参数2差值', f'{safe_name}_部分池化预测',
                           f'{safe_name}_部分池化vs参数1差值', f'{safe_name}_部分池化vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    hyperparam_rows = [
        '【汇总】全局_信号方差',
        '【汇总】全局_长度尺度',
        '【汇总】全局_噪声方差',
        '【汇总】全局_log边际似然',
    ]
    hp_keys = ['signal_variance', 'length_scale', 'noise_variance', 'log_marginal_likelihood']

    for row_name, hp_key in zip(hyperparam_rows, hp_keys):
        row = {'部门名称': row_name, '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2_pp):
            safe_name = combo_name.replace('+', '_')
            val = pp_results[combo_name]['global_params'].get(hp_key, '')
            if isinstance(val, float):
                val = round(val, 6)
            for suffix in [f'{safe_name}_全局预测', f'{safe_name}_全局vs参数1差值', 
                           f'{safe_name}_全局vs参数2差值', f'{safe_name}_部分池化预测',
                           f'{safe_name}_部分池化vs参数1差值', f'{safe_name}_部分池化vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    all_cats = set()
    for combo_name, _ in best3_mode2_pp:
        all_cats.update(pp_results[combo_name]['lambdas'].keys())

    for cat in sorted(all_cats):
        row = {'部门名称': f'【汇总】{cat}_收缩权重', '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2_pp):
            safe_name = combo_name.replace('+', '_')
            val = pp_results[combo_name]['lambdas'].get(cat, '')
            if isinstance(val, float):
                val = round(val, 4)
            for suffix in [f'{safe_name}_全局预测', f'{safe_name}_全局vs参数1差值', 
                           f'{safe_name}_全局vs参数2差值', f'{safe_name}_部分池化预测',
                           f'{safe_name}_部分池化vs参数1差值', f'{safe_name}_部分池化vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    for cat in sorted(all_cats):
        for hp_name, hp_key in zip(['信号方差', '长度尺度', '噪声方差'], 
                                    ['signal_variance', 'length_scale', 'noise_variance']):
            row = {'部门名称': f'【汇总】{cat}_{hp_name}', '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
            for i, (combo_name, _) in enumerate(best3_mode2_pp):
                safe_name = combo_name.replace('+', '_')
                gp = pp_results[combo_name]['group_params'].get(cat)
                if gp is not None:
                    val = gp.get(hp_key, '')
                    if isinstance(val, float):
                        val = round(val, 6)
                else:
                    val = 'N/A'
                for suffix in [f'{safe_name}_全局预测', f'{safe_name}_全局vs参数1差值', 
                               f'{safe_name}_全局vs参数2差值', f'{safe_name}_部分池化预测',
                               f'{safe_name}_部分池化vs参数1差值', f'{safe_name}_部分池化vs参数2差值']:
                    row[suffix] = val if '预测' in suffix else ''
            metric_rows.append(row)

    for row in metric_rows:
        summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)

    summary_df.to_csv(os.path.join(BASE_DIR, 'summary_diff_category_pp_26self_best3.csv'), index=False, encoding='utf-8-sig')
    print("  [保存] summary_diff_category_pp_26self_best3.csv")

    # ==================== 生成对比图 ====================
    print("\n[保存] 生成对比图...")

    short_names = []
    for name in dept_names:
        if '（' in name:
            short = name.split('（')[0]
        elif '，' in name:
            short = name.split('，')[0]
        else:
            short = name[:6]
        short_names.append(short)

    x = np.arange(len(dept_names))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    best1_name = best3_mode2_pp[0][0]
    best1_pred_global = pred_25to26_global[best1_name]
    best1_pred_pp = pred_25to26_partial[best1_name]

    ax.plot(x, best1_pred_global, 'o-', label=f'全局GPR: {best1_name}', color='#e74c3c',
            alpha=0.7, markersize=3)
    ax.plot(x, best1_pred_pp, 's-', label=f'部分池化: {best1_name}', color='#3498db',
            alpha=0.7, markersize=3)
    op1_plot = official_param1.copy()
    ax.plot(x, op1_plot, 'k--', label='官方参数1', linewidth=2)
    ax.set_ylabel('编制人数（含组织员）')
    ax.set_title(f'差分模式二最优组合：全局GPR vs 部分池化\n{best1_name}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    mask_plot = ~np.isnan(official_param1)
    max_val = max(official_param1[mask_plot].max(), best1_pred_pp[mask_plot].max()) * 1.1
    ax.scatter(official_param1[mask_plot], best1_pred_pp[mask_plot], c='#3498db', alpha=0.7, s=60,
               edgecolors='white', zorder=3, label='部分池化')
    ax.scatter(official_param1[mask_plot], best1_pred_global[mask_plot], c='#e74c3c', alpha=0.7, s=60,
               edgecolors='white', zorder=3, label='全局GPR', marker='s')
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x', zorder=1)
    ax.set_xlabel('官方参数1', fontsize=10)
    ax.set_ylabel('差分GPR预测', fontsize=10)
    ax.set_title(f'散点图对比：全局GPR MAE={metrics_25to26_global[best1_name]["mae_off1"]:.3f} / '
                 f'部分池化 MAE={metrics_25to26_partial[best1_name]["mae_off1"]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    plt.suptitle('差分GPR 模式二（25→26）最优组合：全局 vs 部分池化',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'gpr_diff_category_pp_26self_best3_analysis.png'), dpi=200, bbox_inches='tight')
    print("  [保存] gpr_diff_category_pp_26self_best3_analysis.png")

    print("\n" + "=" * 80)
    print("[完成] 所有任务执行完毕")
    print("=" * 80)


if __name__ == '__main__':
    main()
