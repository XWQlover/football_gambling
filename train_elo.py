import pandas as pd
from config import *
from dataset.dataset import MyDataset2
from sklearn.metrics import accuracy_score, classification_report
import pickle
import math
from tqdm import tqdm
import os
# 配置参数
config = {
    'data_path': 'data/processed/competition.csv',
    'vocab_file': 'data/processed/team_ids',
    'save_dir': 'checkpoints',
    'checkpoint_path':'checkpoints/'
}

def calculate_elo_with_draw(
        player1_rating: float,
        player2_rating: float,
        result: int,  # 结果：'p1_win'=玩家1胜, 'p2_win'=玩家2胜, 'draw'=平
        k_factor: int = 32
) -> tuple[float, float]:
    """
    支持胜/负/平的 ELO 分数计算函数（一次性返回双方新分数）

    参数:
        player1_rating: 玩家1的当前 ELO 分数
        player2_rating: 玩家2的当前 ELO 分数
        result: 比赛结果，可选值：
                'p1_win'（玩家1胜）、'p2_win'（玩家2胜）、'draw'（平局）
        k_factor: 分数波动系数（默认32，新手可设50，高分段设20）

    返回:
        tuple: (player1_new_rating, player2_new_rating) → 玩家1、玩家2更新后的分数（保留1位小数）
    """

    # 步骤2：计算双方的预期胜率（ELO 核心公式）
    # 玩家1预期胜率 = 1 / (1 + 10^((玩家2分数 - 玩家1分数)/400))
    p1_expected = 1 / (1 + math.pow(10, (player2_rating - player1_rating) / 400))
    # 玩家2预期胜率 = 1 - 玩家1预期胜率（总和为1）
    p2_expected = 1 - p1_expected

    # 步骤3：根据结果定义双方的实际结果
    if result == 2:
        p1_actual = 1.0
        p2_actual = 0.0
    elif result == 0:
        p1_actual = 0.0
        p2_actual = 1.0
    else:  # draw
        p1_actual = 0.5
        p2_actual = 0.5

    # 步骤4：计算双方分数变化并更新
    p1_change = k_factor * (p1_actual - p1_expected)
    p2_change = k_factor * (p2_actual - p2_expected)

    p1_new = round(player1_rating + p1_change, 1)
    p2_new = round(player2_rating + p2_change, 1)

    return p1_new, p2_new


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
    train_data = data[data["时间"] < val_start_time].copy()
    print(f"训练集数据范围: < {val_start_time}, 样本数: {len(train_data)}")
    train_dataset = MyDataset2(
        data=train_data,
        vocab_file=config['vocab_file'],
        start_time="2022-01-01", end_time=val_start_time,
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
    socre = 1500
    id2score = {k:socre for k,v in train_dataset.id2team.items()}
    # 6. 训练循环
    print("=" * 50)
    print("开始训练...")
    for sample in train_dataset:
        a,b = calculate_elo_with_draw(
                                id2score[sample["team_a_id"]],
                                id2score[sample["team_b_id"]],
                                sample["labels"]
                                )
        id2score[sample["team_a_id"]], id2score[sample["team_2_id"]] = a,b
    preds,labels = [],[]
    for sample in val_dataset:
        pred = 2 if id2score[sample["team_a_id"]] > id2score[sample["team_b_id"]] \
                else ( 1 if id2score[sample["team_a_id"]] == id2score[sample["team_b_id"]] else 0)
        label = sample["labels"]
        preds.append(pred)
        labels.append(label)
    val_acc = accuracy_score(preds,labels)
    pickle.dump(id2score,open(f"{config['save_dir']}/id2score.pkl","wb"))
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
    id2score = pickle.load(open(f"{config['save_dir']}/id2score.pkl","rb"))

    preds, labels = [], []
    for sample in test_dataset:
        pred = 2 if id2score[sample["team_a_id"]] > id2score[sample["team_b_id"]] \
            else (1 if id2score[sample["team_a_id"]] == id2score[sample["team_b_id"]] else 0)
        label = sample["labels"]
        preds.append(pred)
        labels.append(label)
    # 5. 详细分类报告
    print("\n" + "=" * 50)
    print("分类报告:")
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
    else:
        test()
