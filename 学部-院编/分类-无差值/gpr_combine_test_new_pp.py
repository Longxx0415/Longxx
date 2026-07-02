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
TRAIN_2025_PATH = os.path.join(BASE_DIR, 'data2025.csv')
TRAIN_2026_PATH = os.path.join(BASE_DIR, 'data2026.csv')
OFFICIAL_OLS_XLSX = os.path.join(BASE_DIR, '测算数据（两个参数）_backup.xlsx')

# 用于判断特征是否全为0的完整特征列（保持不变）
FEATURE_COLS_ALL = ['本科生', '研究生', '留学生人数', '长期留学生人数', '折合学生数',
                      '在编-教师', '公共课学分', '公共课数量', '是否公共课']

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


def identify_zero_feature_depts(df, feature_cols):
    """识别所有特征值均为0的学院"""
    mask = (df[feature_cols] == 0).all(axis=1)
    return df.loc[mask, '部门名称'].tolist()


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
    """
    pred_total: 含组织员的预测编制人数
    org_count: 组织员人数
    返回：正科、副科、科级以下
    """
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
    """计算与官方参数的MAE和RMSE，自动排除NaN值（未匹配到的学院不参与计算）"""
    valid_mask = ~np.isnan(official_vals)
    if np.sum(valid_mask) == 0:
        return np.nan, np.nan
    diff = y_pred[valid_mask] - official_vals[valid_mask]
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    return mae, rmse


def extract_gpr_params(gpr):
    """提取GPR训练后的超参数"""
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
    except Exception as e:
        return {
            'signal_variance': np.nan,
            'length_scale': np.nan,
            'noise_variance': np.nan,
            'log_marginal_likelihood': np.nan,
            'alpha': gpr.alpha
        }


def print_gpr_params(gpr, combo_name):
    """打印GPR训练后的重要参数"""
    params = extract_gpr_params(gpr)
    print(f"  [GPR参数] {combo_name}:")
    print(f"    信号方差 (signal variance): {params['signal_variance']:.6f}")
    print(f"    长度尺度 (length scale): {params['length_scale']:.6f}")
    print(f"    噪声方差 (noise variance): {params['noise_variance']:.6f}")
    print(f"    alpha (正则化): {params['alpha']:.6f}")
    print(f"    log边际似然: {params['log_marginal_likelihood']:.4f}")


# ==================== 部分池化预测 ====================
def partial_pooling_predict(df_train, gpr_idx, zero_idx, feature_cols, dept_names, n_depts):
    """
    按大类进行部分池化预测
    返回：全局预测、部分池化预测、全局超参数、各组超参数、收缩权重
    """
    # 全局GPR
    X_train = df_train.loc[gpr_idx, feature_cols].values.astype(float)
    y_train = df_train.loc[gpr_idx, '编制人数'].values.astype(float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    gpr_global = GaussianProcessRegressor(
        kernel=KERNEL, n_restarts_optimizer=10, normalize_y=True,
        alpha=1e-2, random_state=RANDOM_STATE
    )
    gpr_global.fit(X_train_scaled, y_train)
    global_params = extract_gpr_params(gpr_global)

    # 全局预测
    y_global = np.zeros(n_depts)
    y_global[zero_idx] = df_train.loc[zero_idx, '编制人数'].values.astype(float)
    y_pred_global, _ = gpr_global.predict(X_train_scaled, return_std=True)
    y_pred_global = np.maximum(y_pred_global, 0)
    y_global[gpr_idx] = y_pred_global

    # 获取大类
    categories = df_train.loc[gpr_idx, '大类'].values
    unique_cats = np.unique(categories)
    K = len(unique_cats)
    n_global = len(gpr_idx)
    n_bar = n_global / K

    # 分组GPR
    group_params = {}
    group_preds = {}
    lambdas = {}

    for cat in unique_cats:
        cat_mask = categories == cat
        cat_idx_in_gpr = np.where(cat_mask)[0]
        cat_global_idx = [gpr_idx[i] for i in cat_idx_in_gpr]
        n_k = len(cat_idx_in_gpr)

        if n_k < 2:
            # 样本太少，使用全局
            lambdas[cat] = 0.0
            group_preds[cat] = y_global.copy()
            group_params[cat] = global_params.copy()
            continue

        X_train_cat = df_train.loc[cat_global_idx, feature_cols].values.astype(float)
        y_train_cat = df_train.loc[cat_global_idx, '编制人数'].values.astype(float)
        X_train_cat_scaled = scaler.transform(X_train_cat)

        gpr_cat = GaussianProcessRegressor(
            kernel=KERNEL, n_restarts_optimizer=10, normalize_y=True,
            alpha=1e-2, random_state=RANDOM_STATE
        )
        gpr_cat.fit(X_train_cat_scaled, y_train_cat)
        group_params[cat] = extract_gpr_params(gpr_cat)

        # 对该组学院进行预测
        y_pred_cat, _ = gpr_cat.predict(X_train_cat_scaled, return_std=True)
        y_pred_cat = np.maximum(y_pred_cat, 0)

        y_cat_full = np.zeros(n_depts)
        y_cat_full[zero_idx] = df_train.loc[zero_idx, '编制人数'].values.astype(float)
        for j, idx in enumerate(cat_global_idx):
            y_cat_full[idx] = y_pred_cat[j]
        group_preds[cat] = y_cat_full

        # 收缩权重：λ = n_k / (n_k + n̄)
        lambdas[cat] = n_k / (n_k + n_bar)

    # 部分池化预测
    y_partial = np.zeros(n_depts)
    y_partial[zero_idx] = df_train.loc[zero_idx, '编制人数'].values.astype(float)

    for i in gpr_idx:
        cat = df_train.loc[i, '大类']
        lam = lambdas.get(cat, 0.0)
        y_partial[i] = lam * group_preds[cat][i] + (1 - lam) * y_global[i]

    return y_global, y_partial, global_params, group_params, lambdas, scaler


# ==================== 生成特征组合 ====================
def generate_feature_combos():
    """
    基础组合（3种）：去掉折合学生数，保留本+研+师及其留学生扩展
    公共课选项（3种）：去掉公共课学分，保留空、公课数量、是否公课
    新增特征组（7个开关，按组约束）：
      - 四个在编（同时）
      - 博士后（独立）
      - 三个派遣（同时）
      - 可上报高研院+不可上报高研院（同时）
      - 可上报柔性+不可上报柔性（同时）
      - 劳动合同制专职科研人员（独立）
      - 一年制科研助理（独立）
    双轨制：完全不考虑
    """
    bases = [
        {'name': '本+研+师', 'cols': ['本科生', '研究生', '在编-教师']},
        {'name': '本+研+师+留', 'cols': ['本科生', '研究生', '在编-教师', '留学生人数']},
        {'name': '本+研+师+长留', 'cols': ['本科生', '研究生', '在编-教师', '长期留学生人数']},
    ]

    public_options = [
        {'name': '', 'cols': []},
        {'name': '+公课数量', 'cols': ['公共课数量']},
        {'name': '+是否公课', 'cols': ['是否公共课']},
    ]

    # 新增特征组（7个开关，按组约束）
    new_groups = [
        {'name': '+在编', 'cols': ['在编-教辅', '在编-思政', '在编-管理', '在编-工勤']},
        {'name': '+博士后', 'cols': ['博士后']},
        {'name': '+派遣', 'cols': ['学校经费派遣', '部门经费派遣', '项目经费派遣']},
        {'name': '+高研院', 'cols': ['可上报高研院', '不可上报高研院']},
        {'name': '+柔性', 'cols': ['可上报柔性', '不可上报柔性']},
        {'name': '+劳动合同制', 'cols': ['劳动合同制专职科研人员']},
        {'name': '+一年制', 'cols': ['一年制科研助理']},
    ]

    combos = []
    for base in bases:
        for pub in public_options:
            # 生成new_groups的所有子集（2^7 = 128种）
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
    print(f"[信息] 共生成 {len(COMBOS)} 种特征组合")

    # 读取官方OLS数据（仅backup）
    df_official = pd.read_excel(OFFICIAL_OLS_XLSX, sheet_name='含组织员')

    # 读取各年数据
    df_train_25 = load_data(TRAIN_2025_PATH)
    df_train_26 = load_data(TRAIN_2026_PATH)

    # 统一学院顺序（以2025年为准）
    dept_names = df_train_25['部门名称'].tolist()
    df_train_26 = df_train_26.set_index('部门名称').reindex(dept_names).reset_index()

    # 识别特征全为0的学院
    zero_depts_25 = identify_zero_feature_depts(df_train_25, FEATURE_COLS_ALL)
    zero_depts_26 = identify_zero_feature_depts(df_train_26, FEATURE_COLS_ALL)
    print(f"[信息] 2025年特征全为0的学院（不参与GPR）: {zero_depts_25}")
    print(f"[信息] 2026年特征全为0的学院（不参与GPR）: {zero_depts_26}")

    # 获取官方拟合值
    official_param1, official_param2 = match_official_ols(dept_names, df_official)

    # 获取组织员人数和实际编制
    org_counts_26 = df_train_26['组织员人数'].values.astype(int)
    y_actual_26 = df_train_26['编制人数'].values.astype(float)

    n_depts = len(dept_names)

    # 区分GPR学院和全0学院
    gpr_mask_25 = ~df_train_25['部门名称'].isin(zero_depts_25)
    gpr_idx_25 = df_train_25[gpr_mask_25].index.tolist()
    zero_idx_25 = df_train_25[~gpr_mask_25].index.tolist()

    gpr_mask_26 = ~df_train_26['部门名称'].isin(zero_depts_26)
    gpr_idx_26 = df_train_26[gpr_mask_26].index.tolist()
    zero_idx_26 = df_train_26[~gpr_mask_26].index.tolist()

    # 存储结果
    pred_25to26 = {}
    metrics_25to26 = {}

    pred_26self = {}
    std_26self = {}
    metrics_26self = {}

    # ==================== 模式1：25→26 验证 ====================
    print("\n" + "=" * 80)
    print(f">>> 模式一：2025年训练 → 2026年验证（共{len(COMBOS)}种组合）")
    print("=" * 80)

    for combo_idx, combo in enumerate(COMBOS):
        combo_name = combo['name']
        feature_cols = combo['cols']

        missing_cols = [c for c in feature_cols if c not in df_train_25.columns]
        if missing_cols:
            continue

        y_pred_26 = np.zeros(n_depts)
        y_pred_26[zero_idx_25] = df_train_25.loc[zero_idx_25, '编制人数'].values.astype(float)

        if len(gpr_idx_25) > 0:
            X_train = df_train_25.loc[gpr_idx_25, feature_cols].values.astype(float)
            y_train = df_train_25.loc[gpr_idx_25, '编制人数'].values.astype(float)
            X_val = df_train_26.loc[gpr_idx_25, feature_cols].values.astype(float)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            gpr = GaussianProcessRegressor(
                kernel=KERNEL, n_restarts_optimizer=10, normalize_y=True,
                alpha=1e-2, random_state=RANDOM_STATE
            )
            gpr.fit(X_train_scaled, y_train)
            y_pred_gpr, _ = gpr.predict(X_val_scaled, return_std=True)
            y_pred_gpr = np.maximum(y_pred_gpr, 0)
            y_pred_26[gpr_idx_25] = y_pred_gpr

        pred_25to26[combo_name] = y_pred_26.copy()

        metrics_actual = evaluate_metrics(y_pred_26, y_actual_26)
        mae_off1, rmse_off1 = calc_mae_rmse_vs_official(y_pred_26, official_param1)

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
        }

        if (combo_idx + 1) % 200 == 0 or combo_idx == len(COMBOS) - 1:
            print(f"  进度: {combo_idx + 1}/{len(COMBOS)}")

    # ==================== 模式2：26→26 自预测 ====================
    print("\n" + "=" * 80)
    print(f">>> 模式二：2026年训练 → 2026年自预测（共{len(COMBOS)}种组合）")
    print("=" * 80)

    for combo_idx, combo in enumerate(COMBOS):
        combo_name = combo['name']
        feature_cols = combo['cols']

        missing_cols = [c for c in feature_cols if c not in df_train_26.columns]
        if missing_cols:
            continue

        y_pred_26_self = np.zeros(n_depts)
        y_std_26_self = np.zeros(n_depts)

        y_pred_26_self[zero_idx_26] = df_train_26.loc[zero_idx_26, '编制人数'].values.astype(float)

        if len(gpr_idx_26) > 0:
            X_train = df_train_26.loc[gpr_idx_26, feature_cols].values.astype(float)
            y_train = df_train_26.loc[gpr_idx_26, '编制人数'].values.astype(float)
            X_pred = df_train_26.loc[gpr_idx_26, feature_cols].values.astype(float)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_pred_scaled = scaler.transform(X_pred)

            gpr = GaussianProcessRegressor(
                kernel=KERNEL, n_restarts_optimizer=10, normalize_y=True,
                alpha=1e-2, random_state=RANDOM_STATE
            )
            gpr.fit(X_train_scaled, y_train)
            y_pred_gpr, y_std_gpr = gpr.predict(X_pred_scaled, return_std=True)
            y_pred_gpr = np.maximum(y_pred_gpr, 0)
            y_pred_26_self[gpr_idx_26] = y_pred_gpr
            y_std_26_self[gpr_idx_26] = y_std_gpr

        pred_26self[combo_name] = y_pred_26_self.copy()
        std_26self[combo_name] = y_std_26_self.copy()

        metrics_actual = evaluate_metrics(y_pred_26_self, y_actual_26)
        mae_off1, rmse_off1 = calc_mae_rmse_vs_official(y_pred_26_self, official_param1)
        mae_off2, rmse_off2 = calc_mae_rmse_vs_official(y_pred_26_self, official_param2)

        metrics_26self[combo_name] = {
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

        if (combo_idx + 1) % 200 == 0 or combo_idx == len(COMBOS) - 1:
            print(f"  进度: {combo_idx + 1}/{len(COMBOS)}")

    # ==================== 找出最优3个组合（按MAE_vs参数1排序）====================
    sorted_mode1 = sorted(metrics_25to26.items(), key=lambda x: x[1]['mae_off1'])
    best3_mode1 = sorted_mode1[:3]

    sorted_mode2 = sorted(metrics_26self.items(), key=lambda x: x[1]['mae_off1'])
    best3_mode2 = sorted_mode2[:3]

    print("\n" + "=" * 80)
    print(">>> 模式一最优3个组合（按MAE_vs backup参数1排序）")
    print("=" * 80)
    for i, (name, m) in enumerate(best3_mode1):
        print(f"\n  第{i+1}名: {name}")
        print(f"    MAE_vs参数1={m['mae_off1']:.3f}, RMSE_vs参数1={m['rmse_off1']:.3f}")
        print(f"    vs实际: MAE={m['mae_actual']:.3f}, RMSE={m['rmse_actual']:.3f}, R²={m['r2']:.3f}")
        print(f"    均值检验: t_p={m['t_p']:.3f} ({m['mean_conclusion']})")
        print(f"    方差检验: F_p={m['p_var_f']:.3f} ({m['f_conclusion']})")

    print("\n" + "=" * 80)
    print(">>> 模式二最优3个组合（按MAE_vs backup参数1排序）")
    print("=" * 80)
    for i, (name, m) in enumerate(best3_mode2):
        print(f"\n  第{i+1}名: {name}")
        print(f"    MAE_vs参数1={m['mae_off1']:.3f}, RMSE_vs参数1={m['rmse_off1']:.3f}")
        print(f"    MAE_vs参数2={m['mae_off2']:.3f}, RMSE_vs参数2={m['rmse_off2']:.3f}")
        print(f"    vs实际: MAE={m['mae_actual']:.3f}, RMSE={m['rmse_actual']:.3f}, R²={m['r2']:.3f}")
        print(f"    均值检验: t_p={m['t_p']:.3f} ({m['mean_conclusion']})")
        print(f"    方差检验: F_p={m['p_var_f']:.3f} ({m['f_conclusion']})")

    # ==================== 对最优3个组合进行部分池化 ====================
    print("\n" + "=" * 80)
    print(">>> 对模式二最优3个组合进行部分池化（按大类分组）")
    print("=" * 80)

    pp_results = {}  # combo_name -> {y_global, y_partial, global_params, group_params, lambdas}
    pp_metrics = {}  # combo_name -> metrics dict

    for combo_name, _ in best3_mode2:
        feature_cols = [c['cols'] for c in COMBOS if c['name'] == combo_name][0]

        print(f"\n[部分池化] 组合: {combo_name}")

        y_global, y_partial, global_params, group_params, lambdas, _ =             partial_pooling_predict(df_train_26, gpr_idx_26, zero_idx_26, feature_cols, dept_names, n_depts)

        pp_results[combo_name] = {
            'y_global': y_global,
            'y_partial': y_partial,
            'global_params': global_params,
            'group_params': group_params,
            'lambdas': lambdas
        }

        # 打印超参数
        print(f"  [全局超参数]")
        print(f"    信号方差: {global_params['signal_variance']:.6f}")
        print(f"    长度尺度: {global_params['length_scale']:.6f}")
        print(f"    噪声方差: {global_params['noise_variance']:.6f}")
        print(f"    log边际似然: {global_params['log_marginal_likelihood']:.4f}")

        for cat, lam in lambdas.items():
            print(f"  [{cat}] 收缩权重 λ={lam:.4f}")
            if cat in group_params and group_params[cat] is not None:
                gp = group_params[cat]
                print(f"    组信号方差: {gp['signal_variance']:.6f}")
                print(f"    组长度尺度: {gp['length_scale']:.6f}")
                print(f"    组噪声方差: {gp['noise_variance']:.6f}")

        # 评估部分池化效果
        mae_off1_pp, rmse_off1_pp = calc_mae_rmse_vs_official(y_partial, official_param1)
        mae_off2_pp, rmse_off2_pp = calc_mae_rmse_vs_official(y_partial, official_param2)

        pp_metrics[combo_name] = {
            'mae_off1': mae_off1_pp,
            'rmse_off1': rmse_off1_pp,
            'mae_off2': mae_off2_pp,
            'rmse_off2': rmse_off2_pp,
        }

        print(f"  [效果] 部分池化 MAE_vs参数1={mae_off1_pp:.3f}, RMSE_vs参数1={rmse_off1_pp:.3f}")
        print(f"  [对比] 全局GPR MAE_vs参数1={metrics_26self[combo_name]['mae_off1']:.3f}")

    # ==================== 输出模式2最优3个组合的detail（含部分池化）====================
    print("\n[保存] 模式二最优3个组合的detail CSV（含部分池化）...")
    for i, (combo_name, _) in enumerate(best3_mode2):
        pred = pred_26self[combo_name]
        std = std_26self[combo_name]
        y_partial = pp_results[combo_name]['y_partial']

        split = np.zeros((n_depts, 3))
        split_pp = np.zeros((n_depts, 3))
        for j in range(n_depts):
            split[j] = split_levels(pred[j], org_counts_26[j])
            split_pp[j] = split_levels(y_partial[j], org_counts_26[j])

        detail_df = pd.DataFrame({
            '部门名称': dept_names,
            '大类': df_train_26['大类'].values,
            '2026实际编制': y_actual_26.astype(int),
            '全局GPR预测': np.round(pred, 2),
            '部分池化预测': np.round(y_partial, 2),
            'GPR_std': np.round(std, 2),
            '官方参数1_backup': ['null' if pd.isna(v) else round(v, 2) for v in official_param1],
            '官方参数2_backup': ['null' if pd.isna(v) else round(v, 2) for v in official_param2],
            '正科_全局(20%)': np.round(split[:, 0], 2),
            '副科_全局(30%)': np.round(split[:, 1], 2),
            '科级以下_全局(50%)': np.round(split[:, 2], 2),
            '正科_部分池化(20%)': np.round(split_pp[:, 0], 2),
            '副科_部分池化(30%)': np.round(split_pp[:, 1], 2),
            '科级以下_部分池化(50%)': np.round(split_pp[:, 2], 2),
        })
        safe_name = combo_name.replace('+', '_')
        fname = f'gpr_26self_{safe_name}.csv'
        detail_df.to_csv(os.path.join(BASE_DIR, fname), index=False, encoding='utf-8-sig')
        print(f"  [保存] {fname}")

    # ==================== 生成模式2汇总表（含部分池化）====================
    print("\n[保存] 模式二汇总表（含部分池化与超参数）...")
    op1_str = ['null' if pd.isna(v) else str(round(v, 2)) for v in official_param1]
    op2_str = ['null' if pd.isna(v) else str(round(v, 2)) for v in official_param2]

    summary_data = {
        '部门名称': dept_names,
        '大类': df_train_26['大类'].values,
        '官方参数1_backup': op1_str,
        '官方参数2_backup': op2_str,
        '2026实际编制': y_actual_26.astype(int),
    }

    for i, (combo_name, _) in enumerate(best3_mode2):
        safe_name = combo_name.replace('+', '_')
        pred = pred_26self[combo_name]
        y_partial = pp_results[combo_name]['y_partial']
        diff1 = pred - official_param1
        diff2 = pred - official_param2
        diff1_pp = y_partial - official_param1
        diff2_pp = y_partial - official_param2

        summary_data[f'{safe_name}_全局预测'] = np.round(pred, 2)
        summary_data[f'{safe_name}_全局vs参数1差值'] = np.round(diff1, 2)
        summary_data[f'{safe_name}_全局vs参数2差值'] = np.round(diff2, 2)
        summary_data[f'{safe_name}_部分池化预测'] = np.round(y_partial, 2)
        summary_data[f'{safe_name}_部分池化vs参数1差值'] = np.round(diff1_pp, 2)
        summary_data[f'{safe_name}_部分池化vs参数2差值'] = np.round(diff2_pp, 2)

    summary_df = pd.DataFrame(summary_data)

    # 指标汇总行
    metrics_map = {
        '【汇总】全局_R²': ('r2', 'global'),
        '【汇总】全局_MAPE(%)': ('mape', 'global'),
        '【汇总】全局_t_p(均值检验)': ('t_p', 'global'),
        '【汇总】全局_均值检验结论': ('mean_conclusion', 'global'),
        '【汇总】全局_F_p(方差检验)': ('p_var_f', 'global'),
        '【汇总】全局_方差检验结论': ('f_conclusion', 'global'),
        '【汇总】全局_Levene_p': ('p_lev', 'global'),
        '【汇总】全局_Levene检验结论': ('lev_conclusion', 'global'),
        '【汇总】全局_MAE_vs参数1': ('mae_off1', 'global'),
        '【汇总】全局_RMSE_vs参数1': ('rmse_off1', 'global'),
        '【汇总】全局_MAE_vs参数2': ('mae_off2', 'global'),
        '【汇总】全局_RMSE_vs参数2': ('rmse_off2', 'global'),
    }

    metric_rows = []
    for row_name, (metric_key, mode) in metrics_map.items():
        row = {'部门名称': row_name, '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2):
            safe_name = combo_name.replace('+', '_')
            if mode == 'global':
                val = metrics_26self[combo_name].get(metric_key, '')
            else:
                val = pp_metrics[combo_name].get(metric_key, '')
            if isinstance(val, float):
                val = round(val, 3)
            for suffix in [f'{safe_name}_全局预测', f'{safe_name}_全局vs参数1差值', 
                           f'{safe_name}_全局vs参数2差值', f'{safe_name}_部分池化预测',
                           f'{safe_name}_部分池化vs参数1差值', f'{safe_name}_部分池化vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    # 部分池化指标行
    pp_metrics_map = {
        '【汇总】部分池化_MAE_vs参数1': 'mae_off1',
        '【汇总】部分池化_RMSE_vs参数1': 'rmse_off1',
        '【汇总】部分池化_MAE_vs参数2': 'mae_off2',
        '【汇总】部分池化_RMSE_vs参数2': 'rmse_off2',
    }
    for row_name, metric_key in pp_metrics_map.items():
        row = {'部门名称': row_name, '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2):
            safe_name = combo_name.replace('+', '_')
            val = pp_metrics[combo_name].get(metric_key, '')
            if isinstance(val, float):
                val = round(val, 3)
            for suffix in [f'{safe_name}_全局预测', f'{safe_name}_全局vs参数1差值', 
                           f'{safe_name}_全局vs参数2差值', f'{safe_name}_部分池化预测',
                           f'{safe_name}_部分池化vs参数1差值', f'{safe_name}_部分池化vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    # 超参数行
    hyperparam_rows = [
        '【汇总】全局_信号方差',
        '【汇总】全局_长度尺度',
        '【汇总】全局_噪声方差',
        '【汇总】全局_log边际似然',
    ]
    hp_keys = ['signal_variance', 'length_scale', 'noise_variance', 'log_marginal_likelihood']

    for row_name, hp_key in zip(hyperparam_rows, hp_keys):
        row = {'部门名称': row_name, '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2):
            safe_name = combo_name.replace('+', '_')
            val = pp_results[combo_name]['global_params'].get(hp_key, '')
            if isinstance(val, float):
                val = round(val, 6)
            for suffix in [f'{safe_name}_全局预测', f'{safe_name}_全局vs参数1差值', 
                           f'{safe_name}_全局vs参数2差值', f'{safe_name}_部分池化预测',
                           f'{safe_name}_部分池化vs参数1差值', f'{safe_name}_部分池化vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    # 各组收缩权重行
    all_cats = set()
    for combo_name, _ in best3_mode2:
        all_cats.update(pp_results[combo_name]['lambdas'].keys())

    for cat in sorted(all_cats):
        row = {'部门名称': f'【汇总】{cat}_收缩权重', '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
        for i, (combo_name, _) in enumerate(best3_mode2):
            safe_name = combo_name.replace('+', '_')
            val = pp_results[combo_name]['lambdas'].get(cat, '')
            if isinstance(val, float):
                val = round(val, 4)
            for suffix in [f'{safe_name}_全局预测', f'{safe_name}_全局vs参数1差值', 
                           f'{safe_name}_全局vs参数2差值', f'{safe_name}_部分池化预测',
                           f'{safe_name}_部分池化vs参数1差值', f'{safe_name}_部分池化vs参数2差值']:
                row[suffix] = val if '预测' in suffix else ''
        metric_rows.append(row)

    # 各组超参数行
    for cat in sorted(all_cats):
        for hp_name, hp_key in zip(['信号方差', '长度尺度', '噪声方差'], 
                                    ['signal_variance', 'length_scale', 'noise_variance']):
            row = {'部门名称': f'【汇总】{cat}_{hp_name}', '大类': '', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
            for i, (combo_name, _) in enumerate(best3_mode2):
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

    summary_df.to_csv(os.path.join(BASE_DIR, 'summary_26self_best3.csv'), index=False, encoding='utf-8-sig')
    print("  [保存] summary_26self_best3.csv")

    # ==================== 生成对比图（仅模式二）====================
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

    # 图: 模式2最优3个（全局 vs 部分池化）
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 子图1: 折线对比（仅最优1的全局和部分池化）
    ax = axes[0]
    best1_name = best3_mode2[0][0]
    best1_pred = pred_26self[best1_name]
    best1_pp = pp_results[best1_name]['y_partial']

    ax.plot(x, best1_pred, 'o-', label=f'全局GPR: {best1_name}', color='#e74c3c',
            alpha=0.7, markersize=3)
    ax.plot(x, best1_pp, 's-', label=f'部分池化: {best1_name}', color='#3498db',
            alpha=0.7, markersize=3)
    op1_plot = official_param1.copy()
    ax.plot(x, op1_plot, 'k--', label='官方参数1', linewidth=2)
    ax.set_ylabel('编制人数（含组织员）')
    ax.set_title(f'模式二最优组合：全局GPR vs 部分池化\n{best1_name}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    # 子图2: 散点图（部分池化 vs 官方参数1）
    ax = axes[1]
    mask_plot = ~np.isnan(official_param1)
    max_val = max(official_param1[mask_plot].max(), best1_pp[mask_plot].max()) * 1.1
    ax.scatter(official_param1[mask_plot], best1_pp[mask_plot], c='#3498db', alpha=0.7, s=60,
               edgecolors='white', zorder=3, label='部分池化')
    ax.scatter(official_param1[mask_plot], best1_pred[mask_plot], c='#e74c3c', alpha=0.7, s=60,
               edgecolors='white', zorder=3, label='全局GPR', marker='s')
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x', zorder=1)
    ax.set_xlabel('官方参数1', fontsize=10)
    ax.set_ylabel('GPR预测', fontsize=10)
    ax.set_title(f'散点图对比：全局GPR MAE={metrics_26self[best1_name]["mae_off1"]:.3f} / '
                 f'部分池化 MAE={pp_metrics[best1_name]["mae_off1"]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    plt.suptitle('GPR 模式二（26→26）最优组合：全局 vs 部分池化',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'gpr_26self_best3_analysis.png'), dpi=200, bbox_inches='tight')
    print("  [保存] gpr_26self_best3_analysis.png")

    print("\n" + "=" * 80)
    print("[完成] 所有任务执行完毕")
    print("=" * 80)


if __name__ == '__main__':
    main()
