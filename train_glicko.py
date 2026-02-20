import pandas as pd
from config import *
from dataset.dataset import MyDataset2
from sklearn.metrics import accuracy_score, classification_report
import pickle
from tqdm import tqdm
import math
import trueskill
from tqdm import tqdm
import os

# 配置参数
config = {
    'data_path': 'data/processed/competition.csv',
    'vocab_file': 'data/processed/team_ids',
    'save_dir': 'checkpoints',
    'checkpoint_path': 'checkpoints/'
}

ts = trueskill.TrueSkill(draw_probability=0.25)


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
    id2team = {k: ts.create_rating() for k, v in train_dataset.id2team.items()}
    # 6. 训练循环
    print("=" * 50)
    print("开始训练...")
    for epoch in range(6):
        for sample in tqdm(train_dataset):
            ida, idb = sample["team_a_id"], sample["team_b_id"]
            teama, teamb = id2team[ida], id2team[idb]
            rank = None
            if sample["label"] == 2:
                rank = [0, 1]
            elif sample["label"] == 1:
                rank = [0, 0]
            elif sample["label"] == 0:
                rank = [1, 0]
            new_a1, new_b1 = ts.rate([[teama], [teamb]], ranks=rank)
            id2team[ida], id2team[idb] = new_a1[0], new_b1[0]

    preds, labels = [], []
    for sample in tqdm(val_dataset):
        ida, idb = sample["team_a_id"], sample["team_b_id"]
        teama, teamb = id2team[ida], id2team[idb]
        k = ts.mu / ts.sigma
        mua = teama.mu
        mub = teamb.mu

        if abs(mua-mub)<=0.1:
            pred = 1
        elif mua > mub:
            pred = 2
        elif mua < mub:
            pred = 0

        label = sample["label"]
        preds.append(pred)
        labels.append(label)

        rank = None
        if sample["label"] == 2:
            rank = [0, 1]
        elif sample["label"] == 1:
            rank = [0, 0]
        elif sample["label"] == 0:
            rank = [1, 0]
        new_a1, new_b1 = ts.rate([[teama], [teamb]], ranks=rank)
        id2team[ida], id2team[idb] = new_a1[0], new_b1[0]

    val_acc = accuracy_score(preds, labels)
    pickle.dump(id2team, open(f"{config['save_dir']}/id2team.pkl", "wb"))
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
    id2team = pickle.load(open(f"{config['save_dir']}/id2team.pkl", "rb"))

    preds, labels = [], []
    for sample in tqdm(test_dataset):
        ida, idb = sample["team_a_id"], sample["team_b_id"]
        teama, teamb = id2team[ida], id2team[idb]
        k = ts.mu / ts.sigma
        mua = teama.mu
        mub = teamb.mu
        if abs(mua-mub)<=0.1:
            pred = 1
        elif mua > mub:
            pred = 2
        elif mua < mub:
            pred = 0

        label = sample["label"]
        preds.append(pred)
        labels.append(label)

        rank = None
        if sample["label"] == 2:
            rank = [0, 1]
        elif sample["label"] == 1:
            rank = [0, 0]
        elif sample["label"] == 0:
            rank = [1, 0]
        new_a1, new_b1 = ts.rate([[teama], [teamb]], ranks=rank)
        id2team[ida], id2team[idb] = new_a1[0], new_b1[0]
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
