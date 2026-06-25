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

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_2025_PATH = os.path.join(BASE_DIR, 'data2025.csv')
TRAIN_2026_PATH = os.path.join(BASE_DIR, 'data2026.csv')
PREDICT_2030_PATH = os.path.join(BASE_DIR, 'forecastdata.csv')
OFFICIAL_OLS_XLSX = os.path.join(BASE_DIR, '测算数据（两个参数）.xlsx')
OFFICIAL_OLS_XLSX_BACKUP = os.path.join(BASE_DIR, '测算数据（两个参数）_backup.xlsx')

# 用于判断特征是否全为0的完整特征列
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
    # 修复MAPE：严格排除实际值为0、nan或inf的项，避免inf
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


def print_gpr_params(gpr, combo_name):
    """打印GPR训练后的重要参数"""
    try:
        sig_var = gpr.kernel_.k1.k1.constant_value
        length_scale = gpr.kernel_.k1.k2.length_scale
        noise_var = gpr.kernel_.k2.noise_level
        print(f"  [GPR参数] {combo_name}:")
        print(f"    信号方差 (signal variance): {sig_var:.6f}")
        print(f"    长度尺度 (length scale): {length_scale:.6f}")
        print(f"    噪声方差 (noise variance): {noise_var:.6f}")
        print(f"    alpha (正则化): {gpr.alpha:.6f}")
        print(f"    log边际似然: {gpr.log_marginal_likelihood_value_:.4f}")
    except Exception as e:
        print(f"  [GPR参数] {combo_name}: 解析参数失败 ({e})")


# ==================== 生成特征组合 ====================
def generate_feature_combos():
    """
    规则：
    1. 本科生+研究生+在编-教师 + (留学生或长期留学生或空) 与 折合学生数 二选一
    2. 公共课学分、公共课数量、是否公共课 最多取一个
    3. 留学生人数 与 长期留学生人数 最多选一个
    """
    public_options = [
        {'name': '', 'cols': []},
        {'name': '+公课学分', 'cols': ['公共课学分']},
        {'name': '+公课数量', 'cols': ['公共课数量']},
        {'name': '+是否公课', 'cols': ['是否公共课']},
    ]

    base = [
        {'name': '本+研+师', 'cols': ['本科生', '研究生', '在编-教师']},
        {'name': '本+研+师+留', 'cols': ['本科生', '研究生', '在编-教师', '留学生人数']},
        {'name': '本+研+师+长留', 'cols': ['本科生', '研究生', '在编-教师', '长期留学生人数']},
        {'name': '折合学生数', 'cols': ['折合学生数']},
    ]

    combos = []
    for b in base:
        for pub in public_options:
            combos.append({
                'name': b['name'] + pub['name'],
                'cols': b['cols'] + pub['cols']
            })
    return combos


# ==================== 主程序 ====================
def main():
    COMBOS = generate_feature_combos()

    # 读取官方OLS数据
    df_official_30 = pd.read_excel(OFFICIAL_OLS_XLSX, sheet_name='含组织员')
    df_official_26 = pd.read_excel(OFFICIAL_OLS_XLSX_BACKUP, sheet_name='含组织员')

    # 读取各年数据
    df_train_25 = load_data(TRAIN_2025_PATH)
    df_train_26 = load_data(TRAIN_2026_PATH)
    df_pred_30 = load_data(PREDICT_2030_PATH)

    # 统一学院顺序（以2025年为准）
    dept_names = df_train_25['部门名称'].tolist()
    df_train_26 = df_train_26.set_index('部门名称').reindex(dept_names).reset_index()
    df_pred_30 = df_pred_30.set_index('部门名称').reindex(dept_names).reset_index()

    # 识别特征全为0的学院
    zero_depts_25 = identify_zero_feature_depts(df_train_25, FEATURE_COLS_ALL)
    zero_depts_26 = identify_zero_feature_depts(df_train_26, FEATURE_COLS_ALL)
    print(f"[信息] 2025年特征全为0的学院（不参与GPR）: {zero_depts_25}")
    print(f"[信息] 2026年特征全为0的学院（不参与GPR）: {zero_depts_26}")

    # 获取官方拟合值
    official_param1_30, _ = match_official_ols(dept_names, df_official_30)
    official_param1_26, official_param2_26 = match_official_ols(dept_names, df_official_26)

    # 获取组织员人数和实际编制
    org_counts_26 = df_train_26['组织员人数'].values.astype(int)
    org_counts_30 = df_pred_30['组织员人数'].values.astype(int)
    y_actual_26 = df_train_26['编制人数'].values.astype(float)

    n_depts = len(dept_names)

    # 存储结果
    metrics_25to26 = {}
    all_pred_30 = {}
    mae_30_vs_official = {}
    all_pred_26_self = {}
    mae_26_self_vs_official = {}
    rmse_26_self_vs_official_p1 = {}  # 模式2 vs 参数1 的 RMSE
    mae_26_self_vs_official_p2 = {}   # 模式2 vs 参数2 的 MAE
    rmse_26_self_vs_official_p2 = {}  # 模式2 vs 参数2 的 RMSE

    # ==================== 模式1：25→26 验证 ====================
    print("\n" + "=" * 80)
    print(">>> 模式一：2025年训练 → 2026年验证")
    print("=" * 80)

    # 区分GPR学院和全0学院
    gpr_mask_25 = ~df_train_25['部门名称'].isin(zero_depts_25)
    gpr_idx_25 = df_train_25[gpr_mask_25].index.tolist()
    zero_idx_25 = df_train_25[~gpr_mask_25].index.tolist()

    for combo_idx, combo in enumerate(COMBOS):
        combo_name = combo['name']
        feature_cols = combo['cols']

        missing_cols = [c for c in feature_cols if c not in df_train_25.columns]
        if missing_cols:
            print(f"\n[跳过] 组合 {combo_name}: 缺少特征列 {missing_cols}")
            continue

        # 初始化预测数组
        y_pred_26 = np.zeros(n_depts)

        # 全0学院：直接拿2025年现值作为2026年预测
        y_pred_26[zero_idx_25] = df_train_25.loc[zero_idx_25, '编制人数'].values.astype(float)

        # GPR学院：正常训练预测
        if len(gpr_idx_25) > 0:
            X_train = df_train_25.loc[gpr_idx_25, feature_cols].values.astype(float)
            y_train = df_train_25.loc[gpr_idx_25, '编制人数'].values.astype(float)
            X_val = df_train_26.loc[gpr_idx_25, feature_cols].values.astype(float)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            gpr = GaussianProcessRegressor(
                kernel=KERNEL, n_restarts_optimizer=50, normalize_y=True,
                alpha=1e-2, random_state=RANDOM_STATE
            )
            gpr.fit(X_train_scaled, y_train)
            y_pred_gpr, _ = gpr.predict(X_val_scaled, return_std=True)
            y_pred_gpr = np.maximum(y_pred_gpr, 0)
            y_pred_26[gpr_idx_25] = y_pred_gpr

            print_gpr_params(gpr, combo_name)

        y_actual = y_actual_26
        metrics_26 = evaluate_metrics(y_pred_26, y_actual)
        mae_off1, rmse_off1 = calc_mae_rmse_vs_official(y_pred_26, official_param1_26)

        metrics_25to26[combo_name] = {
            'mae_actual': metrics_26['mae'], 'rmse_actual': metrics_26['rmse'],
            'r2': metrics_26['r2'], 'mape': metrics_26['mape'],
            't_p': metrics_26['p_mean'], 'mean_conclusion': metrics_26['mean_conclusion'],
            'p_var_f': metrics_26['p_var_f'], 'f_conclusion': metrics_26['f_conclusion'],
            'p_lev': metrics_26['p_lev'], 'lev_conclusion': metrics_26['lev_conclusion'],
            'mae_off1': mae_off1, 'rmse_off1': rmse_off1,
        }

        print(f"\n[{combo_idx + 1}/{len(COMBOS)}] {combo_name}")
        print(f"  vs实际: MAE={metrics_26['mae']:.3f}, RMSE={metrics_26['rmse']:.3f}, "
              f"R²={metrics_26['r2']:.3f}, t_p={metrics_26['p_mean']:.3f}")
        print(f"  vs backup参数1: MAE={mae_off1:.3f}, RMSE={rmse_off1:.3f}")

    # ==================== 模式2：26→26 自预测 ====================
    print("\n" + "=" * 80)
    print(">>> 模式二：2026年训练 → 2026年自预测（与backup官方OLS拟合值比较）")
    print("=" * 80)

    gpr_mask_26 = ~df_train_26['部门名称'].isin(zero_depts_26)
    gpr_idx_26 = df_train_26[gpr_mask_26].index.tolist()
    zero_idx_26 = df_train_26[~gpr_mask_26].index.tolist()

    for combo_idx, combo in enumerate(COMBOS):
        combo_name = combo['name']
        feature_cols = combo['cols']

        missing_cols = [c for c in feature_cols if c not in df_train_26.columns]
        if missing_cols:
            print(f"\n[跳过] 组合 {combo_name}: 缺少特征列 {missing_cols}")
            continue

        y_pred_26_self = np.zeros(n_depts)
        y_std_26_self = np.zeros(n_depts)

        # 全0学院：直接拿2026年现值作为预测
        y_pred_26_self[zero_idx_26] = df_train_26.loc[zero_idx_26, '编制人数'].values.astype(float)

        if len(gpr_idx_26) > 0:
            X_train = df_train_26.loc[gpr_idx_26, feature_cols].values.astype(float)
            y_train = df_train_26.loc[gpr_idx_26, '编制人数'].values.astype(float)
            X_pred = df_train_26.loc[gpr_idx_26, feature_cols].values.astype(float)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_pred_scaled = scaler.transform(X_pred)

            gpr = GaussianProcessRegressor(
                kernel=KERNEL, n_restarts_optimizer=50, normalize_y=True,
                alpha=1e-2, random_state=RANDOM_STATE
            )
            gpr.fit(X_train_scaled, y_train)
            y_pred_gpr, y_std_gpr = gpr.predict(X_pred_scaled, return_std=True)
            y_pred_gpr = np.maximum(y_pred_gpr, 0)
            y_pred_26_self[gpr_idx_26] = y_pred_gpr
            y_std_26_self[gpr_idx_26] = y_std_gpr

            print_gpr_params(gpr, combo_name)

        all_pred_26_self[combo_name] = y_pred_26_self.copy()

        mae_off1, rmse_off1 = calc_mae_rmse_vs_official(y_pred_26_self, official_param1_26)
        mae_26_self_vs_official[combo_name] = mae_off1
        rmse_26_self_vs_official_p1[combo_name] = rmse_off1

        mae_off2, rmse_off2 = calc_mae_rmse_vs_official(y_pred_26_self, official_param2_26)
        mae_26_self_vs_official_p2[combo_name] = mae_off2
        rmse_26_self_vs_official_p2[combo_name] = rmse_off2

        metrics_26_self = evaluate_metrics(y_pred_26_self, y_actual_26)

        # 直接减去组织员，按2:3:5拆分
        split_26 = np.zeros((n_depts, 3))
        for i in range(n_depts):
            split_26[i] = split_levels(y_pred_26_self[i], org_counts_26[i])

        print(f"\n[{combo_idx + 1}/{len(COMBOS)}] {combo_name}")
        print(f"  vs实际: MAE={metrics_26_self['mae']:.3f}, RMSE={metrics_26_self['rmse']:.3f}, "
              f"R²={metrics_26_self['r2']:.3f}, t_p={metrics_26_self['p_mean']:.3f}")
        print(f"  vs backup参数1: MAE={mae_off1:.3f}, RMSE={rmse_off1:.3f}")
        print(f"  vs backup参数2: MAE={mae_off2:.3f}, RMSE={rmse_off2:.3f}")
        print(f"  预测合计(含组织员): {y_pred_26_self.sum():.2f}")

        detail_df = pd.DataFrame({
            '部门名称': dept_names,
            '实际编制': y_actual_26.astype(int),
            'GPR预测(含组织员)': np.round(y_pred_26_self, 2),
            'GPR_std': np.round(y_std_26_self, 2),
            '正科(20%)': np.round(split_26[:, 0], 2),
            '副科(30%)': np.round(split_26[:, 1], 2),
            '科级以下(50%)': np.round(split_26[:, 2], 2),
        })
        detail_df.to_csv(
            os.path.join(BASE_DIR, f'gpr_26_self_detail_{combo_name}.csv'),
            index=False, encoding='utf-8-sig'
        )

    # ==================== 模式3：26→30 预测 ====================
    print("\n" + "=" * 80)
    print(">>> 模式三：2026年训练 → 2030年预测（与非backup官方参数1比较）")
    print("=" * 80)

    for combo_idx, combo in enumerate(COMBOS):
        combo_name = combo['name']
        feature_cols = combo['cols']

        missing_cols = [c for c in feature_cols if c not in df_train_26.columns]
        if missing_cols:
            print(f"\n[跳过] 组合 {combo_name}: 缺少特征列 {missing_cols}")
            continue

        y_pred_30 = np.zeros(n_depts)
        y_std_30 = np.zeros(n_depts)

        # 全0学院：直接拿2026年现值作为2030年预测
        y_pred_30[zero_idx_26] = df_train_26.loc[zero_idx_26, '编制人数'].values.astype(float)

        if len(gpr_idx_26) > 0:
            X_train = df_train_26.loc[gpr_idx_26, feature_cols].values.astype(float)
            y_train = df_train_26.loc[gpr_idx_26, '编制人数'].values.astype(float)
            X_pred = df_pred_30.loc[gpr_idx_26, feature_cols].values.astype(float)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_pred_scaled = scaler.transform(X_pred)

            gpr = GaussianProcessRegressor(
                kernel=KERNEL, n_restarts_optimizer=50, normalize_y=True,
                alpha=1e-2, random_state=RANDOM_STATE
            )
            gpr.fit(X_train_scaled, y_train)
            y_pred_gpr, y_std_gpr = gpr.predict(X_pred_scaled, return_std=True)
            y_pred_gpr = np.maximum(y_pred_gpr, 0)
            y_pred_30[gpr_idx_26] = y_pred_gpr
            y_std_30[gpr_idx_26] = y_std_gpr

            print_gpr_params(gpr, combo_name)

        all_pred_30[combo_name] = y_pred_30.copy()

        mae_off1, rmse_off1 = calc_mae_rmse_vs_official(y_pred_30, official_param1_30)
        mae_30_vs_official[combo_name] = mae_off1

        # 直接减去组织员，按2:3:5拆分
        split_30 = np.zeros((n_depts, 3))
        for i in range(n_depts):
            split_30[i] = split_levels(y_pred_30[i], org_counts_30[i])

        print(f"\n[{combo_idx + 1}/{len(COMBOS)}] {combo_name}")
        print(f"  vs非backup参数1: MAE={mae_off1:.3f}, RMSE={rmse_off1:.3f}")
        print(f"  预测合计(含组织员): {y_pred_30.sum():.2f}")

        detail_df = pd.DataFrame({
            '部门名称': dept_names,
            'GPR预测(含组织员)': np.round(y_pred_30, 2),
            'GPR_std': np.round(y_std_30, 2),
            '正科(20%)': np.round(split_30[:, 0], 2),
            '副科(30%)': np.round(split_30[:, 1], 2),
            '科级以下(50%)': np.round(split_30[:, 2], 2),
        })
        detail_df.to_csv(
            os.path.join(BASE_DIR, f'gpr_30_detail_{combo_name}.csv'),
            index=False, encoding='utf-8-sig'
        )

    # ==================== 生成26自预测汇总表 ====================
    print("\n" + "=" * 80)
    print(">>> 生成26自预测汇总表格（按26自预测MAE_vs backup参数1排序）")
    print("=" * 80)

    sorted_combos_26 = sorted(mae_26_self_vs_official.items(), key=lambda x: x[1])
    sorted_combo_names_26 = [name for name, _ in sorted_combos_26]

    # 官方参数列：nan替换为null字符串，确保所有学院都输出
    op1_26_str = ['null' if pd.isna(v) else str(round(v, 2)) for v in official_param1_26]
    op2_26_str = ['null' if pd.isna(v) else str(round(v, 2)) for v in official_param2_26]

    summary_data_26 = {
        '部门名称': dept_names,
        '官方参数1_backup': op1_26_str,
        '官方参数2_backup': op2_26_str,
        '2026实际编制': y_actual_26.astype(int),
    }

    for combo_name in sorted_combo_names_26:
        pred = all_pred_26_self[combo_name]
        diff = pred - official_param1_26
        summary_data_26[f'{combo_name}_预测'] = np.round(pred, 2)
        summary_data_26[f'{combo_name}_差值'] = np.round(diff, 2)

    summary_df_26 = pd.DataFrame(summary_data_26)

    metric_rows_26 = []

    # R²
    row_r2 = {'部门名称': '【汇总】R²', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        if combo_name in metrics_25to26:
            row_r2[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['r2'], 3)
            row_r2[f'{combo_name}_差值'] = ''
        else:
            row_r2[f'{combo_name}_预测'] = ''
            row_r2[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_r2)

    # MAPE
    row_mape = {'部门名称': '【汇总】MAPE(%)', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        if combo_name in metrics_25to26:
            row_mape[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['mape'], 3)
            row_mape[f'{combo_name}_差值'] = ''
        else:
            row_mape[f'{combo_name}_预测'] = ''
            row_mape[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_mape)

    # t_p
    row_tp = {'部门名称': '【汇总】t_p(均值检验)', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        if combo_name in metrics_25to26:
            row_tp[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['t_p'], 3)
            row_tp[f'{combo_name}_差值'] = ''
        else:
            row_tp[f'{combo_name}_预测'] = ''
            row_tp[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_tp)

    # 均值检验结论
    row_mean = {'部门名称': '【汇总】均值检验结论', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        if combo_name in metrics_25to26:
            row_mean[f'{combo_name}_预测'] = metrics_25to26[combo_name]['mean_conclusion']
            row_mean[f'{combo_name}_差值'] = ''
        else:
            row_mean[f'{combo_name}_预测'] = ''
            row_mean[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_mean)

    # F_p
    row_var = {'部门名称': '【汇总】F_p(方差检验)', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        if combo_name in metrics_25to26:
            row_var[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['p_var_f'], 3)
            row_var[f'{combo_name}_差值'] = ''
        else:
            row_var[f'{combo_name}_预测'] = ''
            row_var[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_var)

    # 方差检验结论
    row_var_c = {'部门名称': '【汇总】方差检验结论', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        if combo_name in metrics_25to26:
            row_var_c[f'{combo_name}_预测'] = metrics_25to26[combo_name]['f_conclusion']
            row_var_c[f'{combo_name}_差值'] = ''
        else:
            row_var_c[f'{combo_name}_预测'] = ''
            row_var_c[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_var_c)

    # Levene_p
    row_lev = {'部门名称': '【汇总】Levene_p', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        if combo_name in metrics_25to26:
            row_lev[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['p_lev'], 3)
            row_lev[f'{combo_name}_差值'] = ''
        else:
            row_lev[f'{combo_name}_预测'] = ''
            row_lev[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_lev)

    # Levene检验结论
    row_lev_c = {'部门名称': '【汇总】Levene检验结论', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        if combo_name in metrics_25to26:
            row_lev_c[f'{combo_name}_预测'] = metrics_25to26[combo_name]['lev_conclusion']
            row_lev_c[f'{combo_name}_差值'] = ''
        else:
            row_lev_c[f'{combo_name}_预测'] = ''
            row_lev_c[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_lev_c)

    # MAE_vs backup参数1（26自预测）
    row_mae_26 = {'部门名称': '【汇总】MAE_vs参数1(26自预测)', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        row_mae_26[f'{combo_name}_预测'] = round(mae_26_self_vs_official[combo_name], 3)
        row_mae_26[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_mae_26)

    # RMSE_vs backup参数1（26自预测）
    row_rmse_26_p1 = {'部门名称': '【汇总】RMSE_vs参数1(26自预测)', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        row_rmse_26_p1[f'{combo_name}_预测'] = round(rmse_26_self_vs_official_p1.get(combo_name, np.nan), 3)
        row_rmse_26_p1[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_rmse_26_p1)

    # MAE_vs backup参数2（26自预测）
    row_mae_26_p2 = {'部门名称': '【汇总】MAE_vs参数2(26自预测)', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        row_mae_26_p2[f'{combo_name}_预测'] = round(mae_26_self_vs_official_p2.get(combo_name, np.nan), 3)
        row_mae_26_p2[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_mae_26_p2)

    # RMSE_vs backup参数2（26自预测）
    row_rmse_26_p2 = {'部门名称': '【汇总】RMSE_vs参数2(26自预测)', '官方参数1_backup': '', '官方参数2_backup': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names_26:
        row_rmse_26_p2[f'{combo_name}_预测'] = round(rmse_26_self_vs_official_p2.get(combo_name, np.nan), 3)
        row_rmse_26_p2[f'{combo_name}_差值'] = ''
    metric_rows_26.append(row_rmse_26_p2)

    for row in metric_rows_26:
        summary_df_26 = pd.concat([summary_df_26, pd.DataFrame([row])], ignore_index=True)

    summary_df_26.to_csv(
        os.path.join(BASE_DIR, 'summary_26_self_prediction.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"[保存] summary_26_self_prediction.csv")

    # ==================== 生成30年预测汇总表 ====================
    print("\n" + "=" * 80)
    print(">>> 生成30年预测汇总表格（按30年MAE_vs非backup参数1排序）")
    print("=" * 80)

    sorted_combos = sorted(mae_30_vs_official.items(), key=lambda x: x[1])
    sorted_combo_names = [name for name, _ in sorted_combos]

    # 官方参数列：nan替换为null字符串
    op1_30_str = ['null' if pd.isna(v) else str(round(v, 2)) for v in official_param1_30]

    summary_data = {
        '部门名称': dept_names,
        '官方参数1': op1_30_str,
        '2026实际编制': y_actual_26.astype(int),
    }

    for combo_name in sorted_combo_names:
        pred = all_pred_30[combo_name]
        diff = pred - official_param1_30
        summary_data[f'{combo_name}_预测'] = np.round(pred, 2)
        summary_data[f'{combo_name}_差值'] = np.round(diff, 2)

    summary_df = pd.DataFrame(summary_data)

    metric_rows = []

    # R²
    row_r2 = {'部门名称': '【汇总】R²', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        if combo_name in metrics_25to26:
            row_r2[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['r2'], 3)
            row_r2[f'{combo_name}_差值'] = ''
        else:
            row_r2[f'{combo_name}_预测'] = ''
            row_r2[f'{combo_name}_差值'] = ''
    metric_rows.append(row_r2)

    # MAPE
    row_mape = {'部门名称': '【汇总】MAPE(%)', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        if combo_name in metrics_25to26:
            row_mape[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['mape'], 3)
            row_mape[f'{combo_name}_差值'] = ''
        else:
            row_mape[f'{combo_name}_预测'] = ''
            row_mape[f'{combo_name}_差值'] = ''
    metric_rows.append(row_mape)

    # t_p
    row_tp = {'部门名称': '【汇总】t_p(均值检验)', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        if combo_name in metrics_25to26:
            row_tp[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['t_p'], 3)
            row_tp[f'{combo_name}_差值'] = ''
        else:
            row_tp[f'{combo_name}_预测'] = ''
            row_tp[f'{combo_name}_差值'] = ''
    metric_rows.append(row_tp)

    # 均值检验结论
    row_mean = {'部门名称': '【汇总】均值检验结论', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        if combo_name in metrics_25to26:
            row_mean[f'{combo_name}_预测'] = metrics_25to26[combo_name]['mean_conclusion']
            row_mean[f'{combo_name}_差值'] = ''
        else:
            row_mean[f'{combo_name}_预测'] = ''
            row_mean[f'{combo_name}_差值'] = ''
    metric_rows.append(row_mean)

    # F_p
    row_var = {'部门名称': '【汇总】F_p(方差检验)', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        if combo_name in metrics_25to26:
            row_var[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['p_var_f'], 3)
            row_var[f'{combo_name}_差值'] = ''
        else:
            row_var[f'{combo_name}_预测'] = ''
            row_var[f'{combo_name}_差值'] = ''
    metric_rows.append(row_var)

    # 方差检验结论
    row_var_c = {'部门名称': '【汇总】方差检验结论', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        if combo_name in metrics_25to26:
            row_var_c[f'{combo_name}_预测'] = metrics_25to26[combo_name]['f_conclusion']
            row_var_c[f'{combo_name}_差值'] = ''
        else:
            row_var_c[f'{combo_name}_预测'] = ''
            row_var_c[f'{combo_name}_差值'] = ''
    metric_rows.append(row_var_c)

    # Levene_p
    row_lev = {'部门名称': '【汇总】Levene_p', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        if combo_name in metrics_25to26:
            row_lev[f'{combo_name}_预测'] = round(metrics_25to26[combo_name]['p_lev'], 3)
            row_lev[f'{combo_name}_差值'] = ''
        else:
            row_lev[f'{combo_name}_预测'] = ''
            row_lev[f'{combo_name}_差值'] = ''
    metric_rows.append(row_lev)

    # Levene检验结论
    row_lev_c = {'部门名称': '【汇总】Levene检验结论', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        if combo_name in metrics_25to26:
            row_lev_c[f'{combo_name}_预测'] = metrics_25to26[combo_name]['lev_conclusion']
            row_lev_c[f'{combo_name}_差值'] = ''
        else:
            row_lev_c[f'{combo_name}_预测'] = ''
            row_lev_c[f'{combo_name}_差值'] = ''
    metric_rows.append(row_lev_c)

    # MAE_vs非backup参数1（30年）
    row_mae_30 = {'部门名称': '【汇总】MAE_vs参数1(30年)', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        row_mae_30[f'{combo_name}_预测'] = round(mae_30_vs_official[combo_name], 3)
        row_mae_30[f'{combo_name}_差值'] = ''
    metric_rows.append(row_mae_30)

    for row in metric_rows:
        summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)

    summary_df.to_csv(
        os.path.join(BASE_DIR, 'summary_30_forecast.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"[保存] summary_30_forecast.csv")

    # ==================== 生成对比图（仅30年预测）====================
    print("\n" + "=" * 80)
    print(">>> 生成对比图（仅30年预测）")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

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

    # 图1: 30年预测 - 各组合 vs 非backup官方参数1
    ax1 = axes[0, 0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_combo_names)))
    for i, combo_name in enumerate(sorted_combo_names):
        if i < 5:
            pred = all_pred_30[combo_name]
            ax1.plot(x, pred, 'o-', label=combo_name, color=colors[i],
                     alpha=0.7, markersize=3)
    # 官方参数1中nan不参与绘图
    op1_plot = official_param1_30.copy()
    ax1.plot(x, op1_plot, 'k--', label='官方参数1', linewidth=2)
    ax1.set_ylabel('编制人数（含组织员）')
    ax1.set_title('2030年预测：各组合预测 vs 官方参数1', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(alpha=0.3)

    # 图2: 30年预测 - 与官方参数1的偏差
    ax2 = axes[0, 1]
    for i, combo_name in enumerate(sorted_combo_names):
        if i < 5:
            pred = all_pred_30[combo_name]
            ax2.plot(x, pred - op1_plot, 'o-', label=combo_name,
                     color=colors[i], alpha=0.7, markersize=3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('偏差（预测 - 官方参数1）')
    ax2.set_title('2030年预测：各组合与官方参数1的偏差', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(alpha=0.3)

    # 图3: MAE排序柱状图
    ax3 = axes[1, 0]
    mae_values = [mae_30_vs_official[name] for name in sorted_combo_names]
    bars = ax3.barh(range(len(sorted_combo_names)), mae_values, color=colors[:len(sorted_combo_names)])
    ax3.set_yticks(range(len(sorted_combo_names)))
    ax3.set_yticklabels(sorted_combo_names, fontsize=7)
    ax3.set_xlabel('MAE_vs参数1')
    ax3.set_title('30年预测各组合MAE排序（从小到大）', fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(alpha=0.3, axis='x')
    for i, v in enumerate(mae_values):
        ax3.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=7)

    # 图4: 最优组合散点图
    ax4 = axes[1, 1]
    best_combo = sorted_combo_names[0]
    best_pred = all_pred_30[best_combo]
    # 过滤掉nan的官方参数1
    mask_plot = ~np.isnan(official_param1_30)
    max_val = max(official_param1_30[mask_plot].max(), best_pred[mask_plot].max()) * 1.1
    ax4.scatter(official_param1_30[mask_plot], best_pred[mask_plot], c='#e74c3c', alpha=0.7, s=60,
               edgecolors='white', zorder=3)
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x', zorder=1)
    ax4.set_xlabel('官方参数1', fontsize=10)
    ax4.set_ylabel('GPR预测', fontsize=10)
    ax4.set_title(f'最优组合散点图：{best_combo}\nMAE={mae_30_vs_official[best_combo]:.3f}',
                  fontsize=10, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max_val)
    ax4.set_ylim(0, max_val)

    plt.suptitle('GPR 2030年预测结果分析',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'gpr_30_forecast_analysis.png'),
                dpi=200, bbox_inches='tight')
    print(f"[保存] gpr_30_forecast_analysis.png")

    # 生成各组合散点图
    n_combos_plot = min(12, len(sorted_combo_names))
    n_rows = (n_combos_plot + 3) // 4
    fig2, axes2 = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes2 = axes2.reshape(1, -1)
    axes2 = axes2.flatten()

    for i, combo_name in enumerate(sorted_combo_names[:n_combos_plot]):
        ax = axes2[i]
        pred = all_pred_30[combo_name]
        mask_plot = ~np.isnan(official_param1_30)
        max_val = max(official_param1_30[mask_plot].max(), pred[mask_plot].max()) * 1.1
        ax.scatter(official_param1_30[mask_plot], pred[mask_plot], c='#e74c3c', alpha=0.7, s=60,
                   edgecolors='white', zorder=3)
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x', zorder=1)
        ax.set_xlabel('官方参数1', fontsize=8)
        ax.set_ylabel('GPR预测', fontsize=8)
        ax.set_title(f'{combo_name}\nMAE={mae_30_vs_official[combo_name]:.3f}',
                     fontsize=8, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

    for j in range(n_combos_plot, len(axes2)):
        axes2[j].set_visible(False)

    plt.suptitle('2030年各组合GPR预测 vs 官方参数1 散点图（按MAE排序）',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'gpr_scatter_all_combos.png'),
                dpi=200, bbox_inches='tight')
    print(f"[保存] gpr_scatter_all_combos.png")

    print("\n" + "=" * 80)
    print("[完成] 所有任务执行完毕")
    print("=" * 80)


if __name__ == '__main__':
    main()