import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
N_CLUSTERS = 3

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


# ==================== 聚类 + 部分池化预测（差分版）====================
def clustering_pooling_predict_diff(df_base, df_target, gpr_idx, zero_idx, feature_cols, dept_names, n_depts, n_clusters=N_CLUSTERS):
    """
    基于差分特征空间进行K-Means聚类，再做部分池化
    返回：部分池化预测差分、全局超参数、各组超参数、收缩权重、聚类标签、聚类中心、scaler
    """
    X_diff = df_target[feature_cols].values.astype(float) - df_base[feature_cols].values.astype(float)
    y_diff = df_target['编制人数'].values.astype(float) - df_base['编制人数'].values.astype(float)

    X_train = X_diff[gpr_idx]
    y_train = y_diff[gpr_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 1. 全局GPR
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

    # 2. 在差分特征空间下聚类
    if len(gpr_idx) < n_clusters * 2:
        return (y_diff_global, global_params, {}, {}, 
                np.zeros(len(gpr_idx), dtype=int), None, scaler)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_train_scaled)
    cluster_centers = km.cluster_centers_

    # 3. 分组GPR + 收缩
    K = n_clusters
    n_global = len(gpr_idx)
    n_bar = n_global / K

    group_params = {}
    group_preds = {}
    lambdas = {}

    for c in range(K):
        cluster_mask = labels == c
        cluster_idx_in_gpr = np.where(cluster_mask)[0]
        cluster_global_idx = [gpr_idx[i] for i in cluster_idx_in_gpr]
        n_k = len(cluster_idx_in_gpr)

        if n_k < 2:
            lambdas[c] = 0.0
            group_preds[c] = y_diff_global.copy()
            group_params[c] = global_params.copy()
            continue

        X_train_c = X_diff[cluster_global_idx]
        y_train_c = y_diff[cluster_global_idx]
        X_train_c_scaled = scaler.transform(X_train_c)

        gpr_c = GaussianProcessRegressor(
            kernel=KERNEL, n_restarts_optimizer=10, normalize_y=True,
            alpha=1e-2, random_state=RANDOM_STATE
        )
        gpr_c.fit(X_train_c_scaled, y_train_c)
        group_params[c] = extract_gpr_params(gpr_c)

        y_pred_c, _ = gpr_c.predict(X_train_c_scaled, return_std=True)
        y_c_full = np.zeros(n_depts)
        y_c_full[zero_idx] = y_diff[zero_idx]
        for j, idx in enumerate(cluster_global_idx):
            y_c_full[idx] = y_pred_c[j]
        group_preds[c] = y_c_full

        lambdas[c] = n_k / (n_k + n_bar)

    # 4. 部分池化预测差分
    y_diff_partial = np.zeros(n_depts)
    y_diff_partial[zero_idx] = y_diff[zero_idx]

    for i in gpr_idx:
        gpr_position = gpr_idx.index(i)
        cluster_label = labels[gpr_position]
        lam = lambdas.get(cluster_label, 0.0)
        y_diff_partial[i] = lam * group_preds[cluster_label][i] + (1 - lam) * y_diff_global[i]

    return y_diff_partial, global_params, group_params, lambdas, labels, cluster_centers, scaler


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
    print(f"[信息] 共生成 {len(COMBOS)} 种特征组合（差分版本，已排除data2024缺失特征），聚类数 K={N_CLUSTERS}")

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

    pred_25to26 = {}
    metrics_25to26 = {}
    params_25to26 = {}
    cluster_info_25to26 = {}

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

    # ==================== 模式2：2025→2026 差分自预测（聚类+部分池化）====================
    print("\n" + "=" * 80)
    print(f">>> 模式二：2025→2026 差分训练 → 2026年自预测（聚类+部分池化，K={N_CLUSTERS}）")
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

        # 聚类 + 部分池化
        y_diff_partial, global_params, group_params, lambdas, labels, centers, _ =             clustering_pooling_predict_diff(df_2025, df_2026, gpr_idx, zero_idx, feature_cols, dept_names, n_depts)

        y_pred_total = base_bianzhi_25 + y_diff_partial
        y_pred_total = np.maximum(y_pred_total, 0)
        pred_25to26[combo_name] = y_pred_total.copy()

        params_25to26[combo_name] = {
            'global': global_params,
            'groups': group_params,
            'lambdas': lambdas
        }

        cluster_labels_full = np.full(n_depts, -1, dtype=int)
        for pos, idx in enumerate(gpr_idx):
            cluster_labels_full[idx] = labels[pos]
        cluster_info_25to26[combo_name] = cluster_labels_full.copy()

        metrics_actual = evaluate_metrics(y_pred_total, y_actual_26)
        mae_off1, rmse_off1 = calc_mae_rmse_vs_official(y_pred_total, official_param1)
        mae_off2, rmse_off2 = calc_mae_rmse_vs_official(y_pred_total, official_param2)

        metrics_25to26[combo_name] = {
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
            'mae_off1': mae_off1,
            'rmse_off1': rmse_off1,
            'mae_off2': mae_off2,
            'rmse_off2': rmse_off2,
        }

        if (combo_idx + 1) % 50 == 0 or combo_idx == len(COMBOS) - 1:
            print(f"  进度: {combo_idx + 1}/{len(COMBOS)}")

    # ==================== 找出最优组合 ====================
    sorted_mode1 = sorted(metrics_24to25.items(), key=lambda x: x[1]['mae_actual'])
    best3_mode1 = sorted_mode1[:3]

    sorted_mode2 = sorted(metrics_25to26.items(), key=lambda x: x[1]['mae_off1'])
    best3_mode2 = sorted_mode2[:3]

    print("\n" + "=" * 80)
    print(">>> 模式一最优3个组合（全局GPR，按MAE_vs实际排序）")
    print("=" * 80)
    for i, (name, m) in enumerate(best3_mode1):
        print(f"\n  第{i+1}名: {name}")
        print(f"    vs实际: MAE={m['mae_actual']:.3f}, RMSE={m['rmse_actual']:.3f}, R²={m['r2']:.3f}")

    print("\n" + "=" * 80)
    print(">>> 模式二最优3个组合（聚类+部分池化，按MAE_vs backup参数1排序）")
    print("=" * 80)
    for i, (name, m) in enumerate(best3_mode2):
        print(f"\n  第{i+1}名: {name}")
        print(f"    特征向量组成: {name}")
        print(f"    MAE_vs参数1={m['mae_off1']:.3f}, RMSE_vs参数1={m['rmse_off1']:.3f}")
        print(f"    MAE_vs参数2={m['mae_off2']:.3f}, RMSE_vs参数2={m['rmse_off2']:.3f}")
        print(f"    vs实际: MAE={m['mae_actual']:.3f}, RMSE={m['rmse_actual']:.3f}, R²={m['r2']:.3f}")
        print(f"    均值检验: t_p={m['t_p']:.3f} ({m['mean_conclusion']})")
        print(f"    方差检验: F_p={m['p_var_f']:.3f} ({m['f_conclusion']})")

    # ==================== 输出模式2最优3个组合的detail ====================
    print("\n[保存] 模式二最优3个组合的detail CSV...")
    for i, (combo_name, _) in enumerate(best3_mode2):
        pred = pred_25to26[combo_name]
        split = np.zeros((n_depts, 3))
        for j in range(n_depts):
            split[j] = split_levels(pred[j], org_counts_26[j])

        labels_full = cluster_info_25to26[combo_name]
        label_str = [f'组{c}' if c >= 0 else '全0' for c in labels_full]

        detail_df = pd.DataFrame({
            '部门名称': dept_names,
            '聚类标签': label_str,
            '2026实际编制': y_actual_26.astype(int),
            '聚类+部分池化预测(含组织员)': np.round(pred, 2),
            '官方参数1_backup': ['null' if pd.isna(v) else round(v, 2) for v in official_param1],
            '官方参数2_backup': ['null' if pd.isna(v) else round(v, 2) for v in official_param2],
            '正科(20%)': np.round(split[:, 0], 2),
            '副科(30%)': np.round(split[:, 1], 2),
            '科级以下(50%)': np.round(split[:, 2], 2),
        })
        safe_name = combo_name.replace('+', '_')
        fname = f'gpr_diff_cluster_pp_26self_{safe_name}.csv'
        detail_df.to_csv(os.path.join(BASE_DIR, fname), index=False, encoding='utf-8-sig')
        print(f"  [保存] {fname}")

    # ==================== 生成模式2汇总表（含超参数与聚类结果）====================
    print("\n[保存] 模式二汇总表（含超参数与聚类结果）...")
    op1_str = ['null' if pd.isna(v) else str(round(v, 2)) for v in official_param1]
    op2_str = ['null' if pd.isna(v) else str(round(v, 2)) for v in official_param2]

    summary_data = {
        '部门名称': dept_names,
        '官方参数1_backup': op1_str,
        '官方参数2_backup': op2_str,
        '2026实际编制': y_actual_26.astype(int),
    }

    for i, (combo_name, _) in enumerate(best3_mode2):
        safe_name = combo_name.replace('+', '_')
        pred = pred_25to26[combo_name]
        diff1 = pred - official_param1
        diff2 = pred - official_param2
        labels_full = cluster_info_25to26[combo_name]
        label_str = [f'组{c}' if c >= 0 else '全0' for c in labels_full]
        summary_data[f'{safe_name}_聚类标签'] = label_str
        summary_data[f'{safe_name}_预测'] = np.round(pred, 2)
        summary_data[f'{safe_name}_vs参数1差值'] = np.round(diff1, 2)
        summary_data[f'{safe_name}_vs参数2差值'] = np.round(diff2, 2)

    summary_df = pd.DataFrame(summary_data)

    metrics_map = {
        '【汇总】R²': 'r2',
        '【汇总】MAPE(%)': 'mape',
        '【汇总】t_p(均值检验)': 't_p',
        '【汇总】均值检验结论': 'mean_conclusion',
        '【汇总】F_p(方差检验)': 'p_var_f',
        '【汇总】方差检验结论': 'f_conclusion',
        '【汇总】Levene_p': 'p_lev',
        '【汇总】Levene检验结论': 'lev_conclusion',
        '【汇总】MAE_vs参数1': 'mae_off1',
        '【汇总】RMSE_vs参数1': 'rmse_off1',
        '【汇总】MAE_vs参数2': 'mae_off2',
        '【汇总】RMSE_vs参数2': 'rmse_off2',
    }

    metric_rows = []
    for row_name, metric_key in metrics_map.items():
        row = {'部门名称': row_name, '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2):
            safe_name = combo_name.replace('+', '_')
            val = metrics_25to26[combo_name].get(metric_key, '')
            if isinstance(val, float):
                val = round(val, 3)
            for suffix in [f'{safe_name}_聚类标签', f'{safe_name}_预测', 
                           f'{safe_name}_vs参数1差值', f'{safe_name}_vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    hp_map = {
        '【汇总】全局_信号方差': 'signal_variance',
        '【汇总】全局_长度尺度': 'length_scale',
        '【汇总】全局_噪声方差': 'noise_variance',
        '【汇总】全局_log边际似然': 'log_marginal_likelihood',
    }
    for row_name, hp_key in hp_map.items():
        row = {'部门名称': row_name, '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2):
            safe_name = combo_name.replace('+', '_')
            val = params_25to26[combo_name]['global'].get(hp_key, '')
            if isinstance(val, float):
                val = round(val, 6)
            for suffix in [f'{safe_name}_聚类标签', f'{safe_name}_预测', 
                           f'{safe_name}_vs参数1差值', f'{safe_name}_vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    for c in range(N_CLUSTERS):
        row = {'部门名称': f'【汇总】组{c}_收缩权重', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2):
            safe_name = combo_name.replace('+', '_')
            val = params_25to26[combo_name]['lambdas'].get(c, '')
            if isinstance(val, float):
                val = round(val, 4)
            for suffix in [f'{safe_name}_聚类标签', f'{safe_name}_预测', 
                           f'{safe_name}_vs参数1差值', f'{safe_name}_vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    for c in range(N_CLUSTERS):
        for hp_name, hp_key in zip(['信号方差', '长度尺度', '噪声方差'], 
                                    ['signal_variance', 'length_scale', 'noise_variance']):
            row = {'部门名称': f'【汇总】组{c}_{hp_name}', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
            for i, (combo_name, _) in enumerate(best3_mode2):
                safe_name = combo_name.replace('+', '_')
                gp = params_25to26[combo_name]['groups'].get(c)
                if gp is not None:
                    val = gp.get(hp_key, '')
                    if isinstance(val, float):
                        val = round(val, 6)
                else:
                    val = 'N/A'
                for suffix in [f'{safe_name}_聚类标签', f'{safe_name}_预测', 
                               f'{safe_name}_vs参数1差值', f'{safe_name}_vs参数2差值']:
                    row[suffix] = val if '预测' in suffix else ''
            metric_rows.append(row)

    for c in range(N_CLUSTERS):
        row = {'部门名称': f'【汇总】组{c}_成员', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2):
            safe_name = combo_name.replace('+', '_')
            labels_full = cluster_info_25to26[combo_name]
            members = [dept_names[j] for j in range(n_depts) if labels_full[j] == c]
            val = '; '.join(members) if members else '无'
            for suffix in [f'{safe_name}_聚类标签', f'{safe_name}_预测', 
                           f'{safe_name}_vs参数1差值', f'{safe_name}_vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    for row in metric_rows:
        summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)

    summary_df.to_csv(os.path.join(BASE_DIR, 'summary_diff_cluster_pp_26self_best3.csv'), index=False, encoding='utf-8-sig')
    print("  [保存] summary_diff_cluster_pp_26self_best3.csv")

    # ==================== 生成对比图（仅模式二最优1）====================
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
    best1_name = best3_mode2[0][0]
    best1_pred = pred_25to26[best1_name]
    ax.plot(x, best1_pred, 'o-', label=f'聚类+部分池化: {best1_name}', color='#3498db',
            alpha=0.7, markersize=3)
    op1_plot = official_param1.copy()
    ax.plot(x, op1_plot, 'k--', label='官方参数1', linewidth=2)
    ax.set_ylabel('编制人数（含组织员）')
    ax.set_title(f'差分模式二最优组合（聚类+部分池化）vs 官方参数1\n{best1_name}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    mask_plot = ~np.isnan(official_param1)
    max_val = max(official_param1[mask_plot].max(), best1_pred[mask_plot].max()) * 1.1
    ax.scatter(official_param1[mask_plot], best1_pred[mask_plot], c='#3498db', alpha=0.7, s=60,
               edgecolors='white', zorder=3)
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x', zorder=1)
    ax.set_xlabel('官方参数1', fontsize=10)
    ax.set_ylabel('聚类+部分池化预测', fontsize=10)
    ax.set_title(f'散点图：{best1_name}\nMAE={metrics_25to26[best1_name]["mae_off1"]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    plt.suptitle('差分GPR 模式二（25→26）聚类+部分池化最优组合分析',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'gpr_diff_cluster_pp_26self_best3_analysis.png'), dpi=200, bbox_inches='tight')
    print("  [保存] gpr_diff_cluster_pp_26self_best3_analysis.png")

    print("\n" + "=" * 80)
    print("[完成] 所有任务执行完毕")
    print("=" * 80)


if __name__ == '__main__':
    main()
