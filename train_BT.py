import numpy as np
from scipy.optimize import minimize
import pandas as pd
from config import *
from dataset.dataset import MyDataset2
from tqdm import tqdm
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report

constraints = [{"type": "ineq", "fun": lambda x: x}]


# ------------------- 步骤2：定义Bradley-Terry损失函数 -------------------
def bt_loss(beta, samples):
    """
    支持胜/负/平的 Bradley-Terry 损失函数（对数似然损失）
    参数：
        beta：各球队的实力值（需要优化的参数，数组形式，索引对应球队ID）
        dataset：比赛数据集，每个样本是字典，包含：
            - "team_a_id"：A队ID（对应beta的索引）
            - "team_b_id"：B队ID（对应beta的索引）
            - "result"：比赛结果（0=A负，1=平局，2=A胜）
            - （可选）"draw_param"：平局参数，默认0.5（可调整）
    返回：
        总损失值（越小越好）
    """
    loss = 0.0
    # 平局参数：θ ∈ (0,1)，值越大平局概率越高（可根据赛事特性调整，默认0.5）
    draw_param = 0.5

    for sample in samples:
        a = sample["team_a_id"]
        b = sample["team_b_id"]
        result = sample["label"]  # 0=A负，1=平，2=A胜
        diff = abs(sample["diff"])
        weight = 1.0
        # 步骤1：计算基础胜率（无平局时A胜/B胜的概率）
        p_a_win = beta[a] / (beta[a] + beta[b])  # A胜概率
        p_b_win = beta[b] / (beta[a] + beta[b])  # B胜概率

        # 步骤2：引入平局，重新分配概率（核心修改）
        # 逻辑：先按比例分配“非平局概率”，剩余部分为平局概率
        # 非平局概率 = 1 - draw_param，平局概率 = draw_param
        p_a_win_with_draw = (1 - draw_param) * p_a_win  # 最终A胜概率
        p_b_win_with_draw = (1 - draw_param) * p_b_win  # 最终B胜概率
        p_draw = draw_param  # 最终平局概率

        # 步骤3：根据实际结果计算对数似然损失
        if result == 2:  # A胜
            loss -= weight * np.log(p_a_win_with_draw + 1e-8)  # 加1e-8避免log(0)
        elif result == 0:  # B胜（A负）
            loss -= weight * np.log(p_b_win_with_draw + 1e-8)
        else:  # 平局
            loss -= weight * np.log(p_draw + 1e-8)

    return loss


def predict_win_prob(idx1, idx2, optimal_beta):
    """预测team1赢team2的概率"""
    prob = optimal_beta[idx1] / (optimal_beta[idx1] + optimal_beta[idx2])
    prob = 2 if prob > 0.5 else (1 if prob == 0.5 else 0)
    return prob


# 配置参数
config = {
    'data_path': 'data/processed/competition.csv',
    'vocab_file': 'data/processed/team_ids',
    'save_dir': 'checkpoints',
    'checkpoint_path': 'checkpoints/'
}


def train():
    """训练主函数"""
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    data = pd.read_csv(config['data_path'])
    data = data.sort_values("时间", ascending=True)

    # 1. 加载数据集（修复数据泄露：只传入对应时间段之前的数据）
    print("=" * 50)
    print("加载数据集...")
    # 训练集：只使用训练时间段之前的数据
    train_data = data[data["时间"] < val_end_time].copy()
    print(f"训练集数据范围: < {val_end_time}, 样本数: {len(train_data)}")
    train_dataset = MyDataset2(
        data=train_data,
        vocab_file=config['vocab_file'],
        start_time="2022-01-01", end_time=val_end_time,
        train=True
    )
    # 验证集：只使用验证时间段之前的数据（包含训练数据，但不包含测试数据）
    val_data = data[data["时间"] < val_end_time].copy()
    print(f"验证集数据范围: < {val_end_time}, 样本数: {len(val_data)}")
    val_dataset = MyDataset2(
        data=val_data,
        vocab_file=config['vocab_file'],
        start_time=val_start_time, end_time=val_end_time,
        train=True
    )
    # 4. 初始化模型
    print("=" * 50)
    print("初始化模型...")
    n_teams = len(train_dataset.id2team)
    # 6. 训练循环
    print("=" * 50)
    print("开始训练...")
    # ------------------- 步骤3：优化求解β值 -------------------
    # 初始值：所有球队β=1（默认实力相等）
    optimal_beta = np.ones(n_teams)
    # 约束：β>0（实力值不能为负）
    # 最小化损失函数，得到最优β值
    samples = []
    teams = set()
    for sample in tqdm(train_dataset):
        a, b = sample["team_a_id"], sample["team_b_id"]
        if (a in teams) or (b in teams):
            result = minimize(bt_loss, optimal_beta, args=(samples), constraints=constraints)
            optimal_beta = result.x

            samples.clear()
            teams.clear()
            samples.append(sample)
            teams.add(a)
            teams.add(b)
        else:
            samples.append(sample)
            teams.add(a)
            teams.add(b)

    samples = []
    teams = set()
    preds, labels = [], []
    for sample in tqdm(val_dataset):
        a, b = sample["team_a_id"], sample["team_b_id"]
        pred = predict_win_prob(a, b, optimal_beta)
        label = sample["label"]
        if (a in teams) or (b in teams):
            result = minimize(bt_loss, optimal_beta, args=(samples), constraints=constraints)
            optimal_beta = result.x

            samples.clear()
            teams.clear()
            samples.append(sample)
            teams.add(a)
            teams.add(b)
        else:
            samples.append(sample)
            teams.add(a)
            teams.add(b)

        preds.append(pred)
        labels.append(label)
    val_acc = accuracy_score(preds, labels)
    pickle.dump(optimal_beta, open(f"{config['save_dir']}/optimal_beta.pkl", "wb"))
    print("=" * 50)
    print("训练完成！")
    print(f"最佳验证准确率: {val_acc:.4f}")


def test():
    """测试主函数"""
    # 配置参数（需要与训练时保持一致）
    data = pd.read_csv(config['data_path'])
    data = data.sort_values("时间", ascending=True)
    # 1. 加载数据集（修复数据泄露：只传入测试时间段之前的数据）
    print("=" * 50)
    print("加载测试数据集...")
    test_dataset = MyDataset2(
        data=data,
        vocab_file=config['vocab_file'],
        start_time=test_start_time, end_time=test_end_time,
        train=True  # 测试时不padding
    )
    # 3. 加载模型
    optimal_beta = pickle.load(open(f"{config['save_dir']}/optimal_beta.pkl", "rb"))
    samples = []
    teams = set()
    preds, labels = [], []
    for sample in tqdm(test_dataset):
        a, b = sample["team_a_id"], sample["team_b_id"]
        pred = predict_win_prob(a, b, optimal_beta)
        label = sample["label"]
        if (a in teams) or (b in teams):
            result = minimize(bt_loss, optimal_beta, args=(samples), constraints=constraints)
            optimal_beta = result.x

            samples.clear()
            teams.clear()
            samples.append(sample)
            teams.add(a)
            teams.add(b)
        else:
            samples.append(sample)
            teams.add(a)
            teams.add(b)

        preds.append(pred)
        labels.append(label)
    # 5. 详细分类报告
    print("\n" + "=" * 50)
    print("分类报告:")
    test_acc = accuracy_score(preds, labels)
    print(f"最佳验证准确率: {test_acc:.4f}")
    print(classification_report(
        labels, preds,
        target_names=['负', '平', '胜'],
        digits=4
    ))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='LSTM模型训练和测试')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式: train 或 test')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
        test()
    else:
        test()
