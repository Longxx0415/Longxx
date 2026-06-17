import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import matplotlib
matplotlib.use('TkAgg')  # 在导入 pyplot 之前切换后端，避开 PyCharm 2024.1 的 bug
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
# 自动定位到本 .py 文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, 'data2026.csv')
PREDICT_PATH = os.path.join(BASE_DIR, 'forecastdata.csv')
OFFICIAL_OLS_PATH = os.path.join(BASE_DIR, '测算数据（两个参数）.xlsx')

EXCLUDE_NAMES = ['上海国际知识产权学院', '中德工程学院，职业技术教育学院']

# 基础特征
BASE_COLS = ['本科生', '研究生', '在编-教师']

# 策略A：基础 + 留学生人数 + 公共课学分
STRATEGY_A_COLS = BASE_COLS + ['留学生人数', '公共课学分']

# 策略B：基础 + 留学生人数 + 公共课数量
STRATEGY_B_COLS = BASE_COLS + ['留学生人数', '公共课数量']

N_COMPONENTS = 4
RANDOM_STATE = 42
RATIOS = np.array([0.2, 0.3, 0.5])
# ================================================


def load_data(path, exclude_names):
    """读取数据并排除指定学院"""
    df = pd.read_csv(path)
    mask = ~df['部门名称'].isin(exclude_names)
    df = df[mask].reset_index(drop=True)
    return df


def build_pca_model(df_train, feature_cols):
    """
    基于指定特征构建PCA模型，计算各学院职能强度指数h*
    """
    scaler = StandardScaler()
    X_train = df_train[feature_cols].values.astype(float)
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    pca.fit(X_train_scaled)
    components = pca.components_
    var_ratio = pca.explained_variance_ratio_

    def compute_H(df):
        """用原始数据直接投影计算综合职能强度指标H"""
        X = df[feature_cols].values.astype(float)
        scores = X @ components.T
        H = scores @ var_ratio
        return H

    # 计算训练年H和h*
    H_train = compute_H(df_train)
    B_train = df_train['编制人数'].values.astype(float)
    h_star = H_train / B_train
    h_star = np.where(np.abs(h_star) < 1e-10, 1e-10, h_star)

    B_total = B_train.sum()
    dept_names = df_train['部门名称'].tolist()
    dept_cats = df_train['大类'].tolist()

    return compute_H, h_star, B_total, dept_names, dept_cats, pca, var_ratio


def solve_optimization(H_pred, h_star, B_total, B_budget):
    """
    求解相对误差优化问题

    目标: min sum((H_i/y_i - h*_i) / h*_i)^2
    约束: sum(y_i) = B_total, 0 < y_i <= B_budget_i
    """
    n = len(H_pred)

    # 保护
    h_star = np.where(np.abs(h_star) < 1e-10, 1e-10, h_star)
    H_pred = np.maximum(H_pred, 1e-10)

    # 目标函数
    def objective(y):
        y = np.maximum(y, 1e-6)
        h = H_pred / y
        return np.sum(((h - h_star) / h_star) ** 2)

    # 等式约束：总编制守恒
    def constraint_sum(y):
        return np.sum(y) - B_total

    # 边界约束
    bounds = [(0.01, B_budget[i]) for i in range(n)]
    constraints = {'type': 'eq', 'fun': constraint_sum}

    # 初始值：均匀分配
    y0 = np.full(n, B_total / n)

    # 求解
    result = minimize(
        objective, y0, method='trust-constr',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 5000, 'gtol': 1e-8, 'xtol': 1e-8}
    )

    y_opt = result.x

    # 后处理：确保严格满足预算约束
    y_opt = np.minimum(y_opt, B_budget)

    # 若预算约束导致总和小于B_total，分配剩余编制到未达上限学院
    current_sum = y_opt.sum()
    if current_sum < B_total - 1e-6:
        remaining = B_total - current_sum
        for _ in range(100):
            if remaining < 1e-6:
                break
            available = [i for i in range(n) if y_opt[i] < B_budget[i] - 0.01]
            if not available:
                break
            adj = remaining / len(available)
            for i in available:
                y_opt[i] = min(y_opt[i] + adj, B_budget[i])
            remaining = B_total - y_opt.sum()

    return y_opt, result


def budget_adjust(total_pred, B_budget):
    """预算约束：超预算则压缩到预算上限，再按2:3:5拆分"""
    total_pred = float(total_pred)
    if total_pred <= B_budget + 1e-9:
        return total_pred, total_pred * RATIOS
    return B_budget, B_budget * RATIOS


def run_strategy(df_train, df_pred, feature_cols, strategy_name):
    """运行单个策略"""
    print(f"\n{'=' * 80}")
    print(f">>> 策略: {strategy_name}")
    print(f"特征: {feature_cols}")
    print(f"{'=' * 80}")

    compute_H, h_star, B_total, dept_names, dept_cats, pca, var_ratio = build_pca_model(df_train, feature_cols)
    n_depts = len(dept_names)

    print(f"学院数量: {n_depts}")
    print(f"2026年总编制: {B_total}")
    print(f"PCA方差贡献率: {var_ratio}")
    print(f"累计贡献率: {var_ratio.sum():.4f}")
    print(f"h* 范围: [{h_star.min():.4f}, {h_star.max():.4f}]")

    H_pred = compute_H(df_pred)
    B_budget = df_pred['预算约束人数'].values.astype(float)

    print(f"H_pred 范围: [{H_pred.min():.2f}, {H_pred.max():.2f}]")
    print(f"预算约束总和: {B_budget.sum()}")

    y_opt, result = solve_optimization(H_pred, h_star, B_total, B_budget)

    print(f"\n优化成功: {result.success}")
    print(f"目标函数值: {result.fun:.6f}")
    print(f"总编制: {y_opt.sum():.2f} (目标: {B_total})")

    # 预算约束与层级拆分
    y_adj_total = np.zeros(n_depts)
    y_adj_split = np.zeros((n_depts, 3))
    for i in range(n_depts):
        adj_total, split = budget_adjust(y_opt[i], B_budget[i])
        y_adj_total[i] = adj_total
        y_adj_split[i] = split

    print(f"约束前总和: {y_opt.sum():.2f}")
    print(f"约束后总和: {y_adj_total.sum():.2f}")
    print(f"顶到预算上限: {np.sum(np.abs(y_adj_total - B_budget) < 0.01)} / {n_depts}")

    return y_opt, y_adj_total, y_adj_split, dept_names, dept_cats, B_budget


# ==================== 主程序 ====================
if __name__ == '__main__':
    df_train = load_data(TRAIN_PATH, EXCLUDE_NAMES)
    df_pred = load_data(PREDICT_PATH, EXCLUDE_NAMES)

    # 读取官方OLS
    df_excel = pd.read_excel(OFFICIAL_OLS_PATH, sheet_name=0)
    df_excel = df_excel[['学院', '拟合数']].copy()
    df_excel = df_excel[~df_excel['学院'].isin(EXCLUDE_NAMES)].reset_index(drop=True)

    # 策略A: 基础+留学生+公共课学分
    y_opt_A, y_adj_A, split_A, names_A, cats_A, budget_A = run_strategy(
        df_train, df_pred, STRATEGY_A_COLS, "策略A: 基础+留学生+公共课学分"
    )

    # 策略B: 基础+留学生+公共课数量
    y_opt_B, y_adj_B, split_B, names_B, cats_B, budget_B = run_strategy(
        df_train, df_pred, STRATEGY_B_COLS, "策略B: 基础+留学生+公共课数量"
    )


    # 官方OLS匹配
    def get_official(dept_names):
        official = []
        for name in dept_names:
            match = df_excel[df_excel['学院'] == name]
            if len(match) > 0:
                official.append(match['拟合数'].values[0])
            else:
                for off_name in df_excel['学院']:
                    if name in off_name or off_name in name:
                        match = df_excel[df_excel['学院'] == off_name]
                        official.append(match['拟合数'].values[0])
                        break
                else:
                    official.append(np.nan)
        return np.array(official)


    official_A = get_official(names_A)
    official_B = get_official(names_B)


    # 评估
    def evaluate(y_opt, official, name):
        mae = np.mean(np.abs(y_opt - official))
        rmse = np.sqrt(np.mean((y_opt - official) ** 2))
        max_err = np.max(np.abs(y_opt - official))
        print(f"\n{name} vs 官方OLS:")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAX: {max_err:.2f}")
        return mae, rmse, max_err


    mae_A, rmse_A, max_A = evaluate(y_opt_A, official_A, "策略A")
    mae_B, rmse_B, max_B = evaluate(y_opt_B, official_B, "策略B")

    # 对比表
    print(f"\n{'=' * 80}")
    print(">>> 两策略对比")
    print(f"{'=' * 80}")
    print(f"{'指标':<<20s} {'策略A(学分)':>15s} {'策略B(数量)':>15s}")
    print("-" * 55)
    print(f"{'MAE':<<20s} {mae_A:>15.3f} {mae_B:>15.3f}")
    print(f"{'RMSE':<<20s} {rmse_A:>15.3f} {rmse_B:>15.3f}")
    print(f"{'MAX':<<20s} {max_A:>15.2f} {max_B:>15.2f}")

    # 详细输出
    print(f"\n{'=' * 80}")
    print(">>> 策略A详细对比")
    print(f"{'=' * 80}")
    print(f"{'序号':<<4s} {'学院':<<28s} {'策略A':>8s} {'官方OLS':>8s} {'偏差':>8s} {'预算':>6s}")
    print("-" * 70)
    for i in range(len(names_A)):
        print(
            f"{i + 1:<4d} {names_A[i][:26]:<<28s} {y_opt_A[i]:8.2f} {official_A[i]:8.2f} {y_opt_A[i] - official_A[i]:8.2f} {budget_A[i]:6.0f}")

    print(f"\n{'=' * 80}")
    print(">>> 策略B详细对比")
    print(f"{'=' * 80}")
    print(f"{'序号':<<4s} {'学院':<<28s} {'策略B':>8s} {'官方OLS':>8s} {'偏差':>8s} {'预算':>6s}")
    print("-" * 70)
    for i in range(len(names_B)):
        print(
            f"{i + 1:<4d} {names_B[i][:26]:<<28s} {y_opt_B[i]:8.2f} {official_B[i]:8.2f} {y_opt_B[i] - official_B[i]:8.2f} {budget_B[i]:6.0f}")

    # 保存
    result_df = pd.DataFrame({
        '大类': cats_A,
        '部门名称': names_A,
        '预算约束': budget_A.astype(int),
        '官方OLS': np.round(official_A, 2),
        '策略A_优化': np.round(y_opt_A, 2),
        '策略A_偏差': np.round(y_opt_A - official_A, 2),
        '策略B_优化': np.round(y_opt_B, 2),
        '策略B_偏差': np.round(y_opt_B - official_B, 2),
    })

    summary = pd.DataFrame([{
        '大类': '合计',
        '部门名称': '-',
        '预算约束': int(budget_A.sum()),
        '官方OLS': np.round(official_A.sum(), 2),
        '策略A_优化': np.round(y_opt_A.sum(), 2),
        '策略A_偏差': np.round(np.sum(y_opt_A - official_A), 2),
        '策略B_优化': np.round(y_opt_B.sum(), 2),
        '策略B_偏差': np.round(np.sum(y_opt_B - official_B), 2),
    }])
    result_df = pd.concat([result_df, summary], ignore_index=True)

    result_df.to_csv('result_two_strategies.csv', index=False, encoding='utf-8-sig')
    print(f"\n[完成] 结果保存至: result_two_strategies.csv")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    short_names = []
    for name in names_A:
        if '（' in name:
            short = name.split('（')[0]
        elif '，' in name:
            short = name.split('，')[0]
        else:
            short = name[:6]
        short_names.append(short)

    x = np.arange(len(names_A))
    width = 0.25

    # 策略A对比
    ax1 = axes[0, 0]
    ax1.bar(x - width, official_A, width, label='官方OLS', color='gray', alpha=0.7)
    ax1.bar(x, y_opt_A, width, label='策略A', color='#3498db', alpha=0.8)
    ax1.set_ylabel('编制人数')
    ax1.set_title(f'策略A vs 官方OLS (MAE={mae_A:.2f})', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[0, 1]
    diff_A = y_opt_A - official_A
    colors_A = ['#3498db' if d > 0 else '#27ae60' for d in diff_A]
    ax2.bar(x, diff_A, color=colors_A, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('偏差')
    ax2.set_title('策略A 偏差分布', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax2.grid(axis='y', alpha=0.3)

    # 策略B对比
    ax3 = axes[1, 0]
    ax3.bar(x - width, official_B, width, label='官方OLS', color='gray', alpha=0.7)
    ax3.bar(x, y_opt_B, width, label='策略B', color='#e67e22', alpha=0.8)
    ax3.set_ylabel('编制人数')
    ax3.set_title(f'策略B vs 官方OLS (MAE={mae_B:.2f})', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    ax4 = axes[1, 1]
    diff_B = y_opt_B - official_B
    colors_B = ['#e67e22' if d > 0 else '#27ae60' for d in diff_B]
    ax4.bar(x, diff_B, color=colors_B, alpha=0.8)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    ax4.set_ylabel('偏差')
    ax4.set_title('策略B 偏差分布', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('两策略预测结果 vs 官方OLS拟合对比', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('compare_two_strategies.png', dpi=200, bbox_inches='tight')

    print(f"[完成] 对比图保存至: compare_two_strategies.png")