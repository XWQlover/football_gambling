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
        diff: int,  # 两队比分差（正数=玩家1净胜，负数=玩家2净胜，0=平局）
        result: int,  # 简化结果：1=玩家1胜，2=玩家2胜，0=平
        k_factor: int = 64
) -> tuple[float, float]:
    """
    支持胜负平+比分差的 ELO 分数计算函数

    参数:
        player1_rating: 玩家1初始ELO分数
        player2_rating: 玩家2初始ELO分数
        diff: 比分差（核心！）
              - 玩家1净胜n球：diff = n（如2-0则diff=2）
              - 玩家2净胜n球：diff = -n（如0-3则diff=-3）
              - 平局：diff = 0
        result: 比赛结果（简化为数字，更易使用）：
                1 = 玩家1胜，2 = 玩家2胜，0 = 平
        k_factor: 基础K因子（默认64，可自定义）

    返回:
        tuple: (player1_new_rating, player2_new_rating) → 双方更新后分数
    """

    # 步骤2：计算双方预期胜率（基础ELO公式）
    p1_expected = 1 / (1 + math.pow(10, (player2_rating - player1_rating) / 400))
    p2_expected = 1 - p1_expected

    # 步骤3：定义比分差调整系数（核心优化点）
    # 规则：比分差越大，调整系数越高，分数变化越明显；设置上限避免异常波动
    def get_diff_coefficient(diff_val: int) -> float:
        abs_diff = abs(diff_val)
        # 分级调整：小胜(1球)=1.0，中胜(2-3球)=1.5，大胜(4+球)=2.0
        if abs_diff == 0:
            return 1.0  # 平局无调整
        else:
            return abs_diff  # 小胜/小负，基础系数


    diff_coeff = get_diff_coefficient(diff)
    print(diff_coeff)
    # 步骤4：结合结果和比分差计算实际结果
    if result == 2:  # 玩家1胜
        p1_actual = 1.0
        p2_actual = 0.0
    elif result == 0:  # 玩家2胜
        p1_actual = 0.0
        p2_actual = 1.0
    else:  # 平局
        p1_actual = 0.5
        p2_actual = 0.5

    # 步骤5：计算分数变化（基础变化 × 比分差调整系数）
    p1_change = k_factor * diff_coeff * (p1_actual - p1_expected)
    p2_change = k_factor * diff_coeff * (p2_actual - p2_expected)

    # 步骤6：更新分数并保留1位小数
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
    socre = 1000
    id2score = {k:socre for k,v in train_dataset.id2team.items()}
    # 6. 训练循环
    print("=" * 50)
    print("开始训练...")
    for sample in train_dataset:
        a,b = calculate_elo_with_draw(
                                id2score[sample["team_a_id"]],
                                id2score[sample["team_b_id"]],
                                sample['diff'],
                                sample["label"]
        )
        id2score[sample["team_a_id"]], id2score[sample["team_b_id"]] = a,b
    preds,labels = [],[]
    for sample in val_dataset:
        pred = 2 if id2score[sample["team_a_id"]] > id2score[sample["team_b_id"]] \
                else ( 1 if id2score[sample["team_a_id"]] == id2score[sample["team_b_id"]] else 0)
        label = sample["label"]
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
        label = sample["label"]
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
