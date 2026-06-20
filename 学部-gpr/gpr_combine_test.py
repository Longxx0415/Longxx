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

EXCLUDE_NAMES = ['上海国际知识产权学院', '中德工程学院，职业技术教育学院']

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
def load_data(path, exclude_names):
    df = pd.read_csv(path, encoding='utf-8-sig')
    mask = ~df['部门名称'].isin(exclude_names)
    return df[mask].reset_index(drop=True)


def match_official_ols(dept_names, df_official):
    """匹配官方OLS拟合值（参数1），支持模糊匹配"""
    official_param1 = []
    for name in dept_names:
        match = df_official[df_official['单位'] == name]
        if len(match) > 0:
            official_param1.append(match['拟合数\n（参数1）'].values[0])
            continue
        for idx, off_name in enumerate(df_official['单位']):
            if pd.isna(off_name):
                continue
            if name in off_name or off_name in name:
                official_param1.append(df_official.iloc[idx]['拟合数\n（参数1）'])
                break
        else:
            official_param1.append(np.nan)
    return np.array(official_param1)


def budget_adjust(total_pred, B_budget, org_count):
    """
    total_pred: 含组织员的预测编制人数
    B_budget: 含组织员的预算约束人数
    org_count: 组织员人数
    返回：约束后含组织员的编制数，以及减去组织员后按2:3:5拆分的三列
    """
    total_pred = float(total_pred)
    B_budget = float(B_budget)
    org_count = int(org_count)

    if total_pred <= B_budget + 1e-9:
        constrained_total = total_pred
    else:
        constrained_total = B_budget

    non_org = max(0, constrained_total - org_count)
    split = non_org * RATIOS
    return constrained_total, split


def evaluate_metrics(y_pred, y_actual):
    n = len(y_actual)
    residual = y_pred - y_actual
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))
    mape = np.mean(np.abs(residual / y_actual)) * 100
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

    # 25->26验证的组合（2025年现在有在编-教师列）
    combos_25to26 = []
    base_25 = [
        {'name': '本+研+师', 'cols': ['本科生', '研究生', '在编-教师']},
        {'name': '本+研+师+留', 'cols': ['本科生', '研究生', '在编-教师', '留学生人数']},
        {'name': '本+研+师+长留', 'cols': ['本科生', '研究生', '在编-教师', '长期留学生人数']},
        {'name': '折合学生数', 'cols': ['折合学生数']},
    ]
    for base in base_25:
        for pub in public_options:
            combos_25to26.append({
                'name': base['name'] + pub['name'],
                'cols': base['cols'] + pub['cols']
            })

    # 26->30预测的组合
    combos_26to30 = []
    base_26 = [
        {'name': '本+研+师', 'cols': ['本科生', '研究生', '在编-教师']},
        {'name': '本+研+师+留', 'cols': ['本科生', '研究生', '在编-教师', '留学生人数']},
        {'name': '本+研+师+长留', 'cols': ['本科生', '研究生', '在编-教师', '长期留学生人数']},
        {'name': '折合学生数', 'cols': ['折合学生数']},
    ]
    for base in base_26:
        for pub in public_options:
            combos_26to30.append({
                'name': base['name'] + pub['name'],
                'cols': base['cols'] + pub['cols']
            })

    return combos_25to26, combos_26to30


# ==================== 主程序 ====================
def main():
    COMBOS_25TO26, COMBOS_26TO30 = generate_feature_combos()

    # 读取官方OLS数据（sheet2：含组织员）
    df_official = pd.read_excel(OFFICIAL_OLS_XLSX, sheet_name='含组织员')

    # 读取各年数据
    df_train_25 = load_data(TRAIN_2025_PATH, EXCLUDE_NAMES)
    df_train_26 = load_data(TRAIN_2026_PATH, EXCLUDE_NAMES)
    df_pred_30 = load_data(PREDICT_2030_PATH, EXCLUDE_NAMES)

    # 统一学院顺序
    dept_names = df_train_25['部门名称'].tolist()
    df_train_26 = df_train_26.set_index('部门名称').reindex(dept_names).reset_index()
    df_pred_30 = df_pred_30.set_index('部门名称').reindex(dept_names).reset_index()

    # 获取官方拟合值（参数1）
    official_param1 = match_official_ols(dept_names, df_official)

    # 获取组织员人数和实际编制
    org_counts_26 = df_train_26['组织员人数'].values.astype(int)
    org_counts_30 = df_pred_30['组织员人数'].values.astype(int)
    y_actual_26 = df_train_26['编制人数'].values.astype(float)
    budget_26 = df_train_26['预算约束人数'].values.astype(float)
    budget_30 = df_pred_30['预算约束人数'].values.astype(float)

    n_depts = len(dept_names)

    # 存储结果
    metrics_25to26 = {}  # 25→26精度指标
    all_pred_30 = {}     # 30年预测（未约束，含组织员）
    mae_30_vs_official = {}  # 30年vs参数1的MAE

    # ==================== 模式1：25→26 验证（仅计算精度指标）====================
    print("=" * 80)
    print(">>> 模式一：2025年训练 → 2026年验证（计算精度指标）")
    print("=" * 80)

    for combo_idx, combo in enumerate(COMBOS_25TO26):
        combo_name = combo['name']
        feature_cols = combo['cols']

        missing_cols = [c for c in feature_cols if c not in df_train_25.columns]
        if missing_cols:
            print(f"\n[跳过] 组合 {combo_name}: 缺少特征列 {missing_cols}")
            continue

        X_train = df_train_25[feature_cols].values.astype(float)
        y_train = df_train_25['编制人数'].values.astype(float)
        X_val = df_train_26[feature_cols].values.astype(float)
        y_actual = df_train_26['编制人数'].values.astype(float)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        gpr = GaussianProcessRegressor(
            kernel=KERNEL, n_restarts_optimizer=50, normalize_y=True,
            alpha=1e-2, random_state=RANDOM_STATE
        )
        gpr.fit(X_train_scaled, y_train)

        y_pred_26, _ = gpr.predict(X_val_scaled, return_std=True)
        y_pred_26 = np.maximum(y_pred_26, 0)

        # 精度指标（vs实际）
        metrics_26 = evaluate_metrics(y_pred_26, y_actual)

        # 与参数1比较（未约束含组织员）
        mae_off1 = np.mean(np.abs(y_pred_26 - official_param1))
        rmse_off1 = np.sqrt(np.mean((y_pred_26 - official_param1) ** 2))

        metrics_25to26[combo_name] = {
            'mae_actual': metrics_26['mae'],
            'rmse_actual': metrics_26['rmse'],
            'r2': metrics_26['r2'],
            'mape': metrics_26['mape'],
            't_p': metrics_26['p_mean'],
            'mean_conclusion': metrics_26['mean_conclusion'],
            'p_var_f': metrics_26['p_var_f'],
            'f_conclusion': metrics_26['f_conclusion'],
            'p_lev': metrics_26['p_lev'],
            'lev_conclusion': metrics_26['lev_conclusion'],
            'mae_off1': mae_off1,
            'rmse_off1': rmse_off1,
        }

        print(f"\n[{combo_idx + 1}/{len(COMBOS_25TO26)}] {combo_name}")
        print(f"  vs实际: MAE={metrics_26['mae']:.3f}, RMSE={metrics_26['rmse']:.3f}, "
              f"R²={metrics_26['r2']:.3f}, t_p={metrics_26['p_mean']:.3f}")
        print(f"  vs参数1: MAE={mae_off1:.3f}, RMSE={rmse_off1:.3f}")

    # ==================== 模式2：26→30 预测 ====================
    print("\n" + "=" * 80)
    print(">>> 模式二：2026年训练 → 2030年预测")
    print("=" * 80)

    for combo_idx, combo in enumerate(COMBOS_26TO30):
        combo_name = combo['name']
        feature_cols = combo['cols']

        missing_cols = [c for c in feature_cols if c not in df_train_26.columns]
        if missing_cols:
            print(f"\n[跳过] 组合 {combo_name}: 缺少特征列 {missing_cols}")
            continue

        X_train = df_train_26[feature_cols].values.astype(float)
        y_train = df_train_26['编制人数'].values.astype(float)
        X_pred = df_pred_30[feature_cols].values.astype(float)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)

        gpr = GaussianProcessRegressor(
            kernel=KERNEL, n_restarts_optimizer=50, normalize_y=True,
            alpha=1e-2, random_state=RANDOM_STATE
        )
        gpr.fit(X_train_scaled, y_train)

        y_pred_30, y_std_30 = gpr.predict(X_pred_scaled, return_std=True)
        y_pred_30 = np.maximum(y_pred_30, 0)
        all_pred_30[combo_name] = y_pred_30.copy()

        # 与参数1比较（未约束含组织员）
        mae_off1 = np.mean(np.abs(y_pred_30 - official_param1))
        rmse_off1 = np.sqrt(np.mean((y_pred_30 - official_param1) ** 2))
        mae_30_vs_official[combo_name] = mae_off1

        # 预算约束 + 减去组织员 + 2:3:5拆分
        adj_total_30 = np.zeros(n_depts)
        adj_split_30 = np.zeros((n_depts, 3))
        for i in range(n_depts):
            adj_total_30[i], adj_split_30[i] = budget_adjust(
                y_pred_30[i], budget_30[i], org_counts_30[i]
            )

        print(f"\n[{combo_idx + 1}/{len(COMBOS_26TO30)}] {combo_name}")
        print(f"  vs参数1: MAE={mae_off1:.3f}, RMSE={rmse_off1:.3f}")
        print(f"  预测合计(含组织员): {y_pred_30.sum():.2f}, 约束后: {adj_total_30.sum():.2f}")

        # 保存该组合的详细结果（约束后）
        detail_df = pd.DataFrame({
            '部门名称': dept_names,
            'GPR预测(含组织员)': np.round(y_pred_30, 2),
            'GPR_std': np.round(y_std_30, 2),
            '预算约束': budget_30.astype(int),
            '约束后总编(含组织员)': np.round(adj_total_30, 2),
            '正科(20%)': np.round(adj_split_30[:, 0], 2),
            '副科(30%)': np.round(adj_split_30[:, 1], 2),
            '科级以下(50%)': np.round(adj_split_30[:, 2], 2),
        })
        detail_df.to_csv(
            os.path.join(BASE_DIR, f'gpr_30_detail_{combo_name}.csv'),
            index=False, encoding='utf-8-sig'
        )

    # ==================== 按MAE排序并生成汇总表 ====================
    print("\n" + "=" * 80)
    print(">>> 生成汇总表格（按30年MAE_vs参数1排序）")
    print("=" * 80)

    # 按30年MAE排序
    sorted_combos = sorted(mae_30_vs_official.items(), key=lambda x: x[1])
    sorted_combo_names = [name for name, _ in sorted_combos]

    # 构建summary表主体（各学院数据）
    summary_data = {
        '部门名称': dept_names,
        '官方参数1': np.round(official_param1, 2),
        '2026实际编制': y_actual_26.astype(int),
    }

    for combo_name in sorted_combo_names:
        pred = all_pred_30[combo_name]
        diff = pred - official_param1
        summary_data[f'{combo_name}_预测'] = np.round(pred, 2)
        summary_data[f'{combo_name}_差值'] = np.round(diff, 2)

    summary_df = pd.DataFrame(summary_data)

    # 在底部添加汇总行（仅保留用户指定的指标）
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

    # MAE_vs参数1（30年）
    row_mae_30 = {'部门名称': '【汇总】MAE_vs参数1(30年)', '官方参数1': '', '2026实际编制': ''}
    for combo_name in sorted_combo_names:
        row_mae_30[f'{combo_name}_预测'] = round(mae_30_vs_official[combo_name], 3)
        row_mae_30[f'{combo_name}_差值'] = ''
    metric_rows.append(row_mae_30)

    # 合并汇总行
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

    # 图1: 30年预测 - 各组合 vs 官方参数1
    ax1 = axes[0, 0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_combo_names)))
    for i, combo_name in enumerate(sorted_combo_names):
        if i < 5:  # 只画前5个
            pred = all_pred_30[combo_name]
            ax1.plot(x, pred, 'o-', label=combo_name, color=colors[i],
                     alpha=0.7, markersize=3)
    ax1.plot(x, official_param1, 'k--', label='官方参数1', linewidth=2)
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
            ax2.plot(x, pred - official_param1, 'o-', label=combo_name,
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

    # 图4: 最优组合散点图（MAE最小的组合）
    ax4 = axes[1, 1]
    best_combo = sorted_combo_names[0]
    best_pred = all_pred_30[best_combo]
    max_val = max(official_param1.max(), best_pred.max()) * 1.1
    ax4.scatter(official_param1, best_pred, c='#e74c3c', alpha=0.7, s=60,
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

    # 生成各组合散点图（按MAE排序）
    n_combos_plot = min(12, len(sorted_combo_names))
    n_rows = (n_combos_plot + 3) // 4
    fig2, axes2 = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes2 = axes2.reshape(1, -1)
    axes2 = axes2.flatten()

    for i, combo_name in enumerate(sorted_combo_names[:n_combos_plot]):
        ax = axes2[i]
        pred = all_pred_30[combo_name]
        max_val = max(official_param1.max(), pred.max()) * 1.1
        ax.scatter(official_param1, pred, c='#e74c3c', alpha=0.7, s=60,
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