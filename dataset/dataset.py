#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import defaultdict



class MyDataset(Dataset):
    """
    LSTM比赛分类Dataset
    输入：原始比赛数据（比赛时间，球队a，球队b，比分）
    输出：适配LSTM的样本（球队A ID、球队B ID、时序特征、标签）
    """

    def __init__(self,
                 data,  # 数据文件路径（csv格式）
                 start_time,
                 end_time,
                 vocab_file="",
                 train=True,
                ):  # 是否为训练集（区分训练/测试）
        self.train = train
        self.start_time = start_time
        self.end_time = end_time
        # 1. 加载并预处理原始数据
        self.raw_data = data
        self.max_len = max(self.raw_data.groupby("主场").count()["id"])

        # 2. 球队名映射为整数ID（适配Embedding层）
        self.team2id, self.id2team = self._build_team_vocab(vocab_file)
        self.num_teams = len(self.team2id)
        # 3. 生成标签（胜/平/负）
        self.raw_data = self._generate_label()
        # 4. 为每个比赛样本构建时序特征序列
        self.processed_data = self._build_sequences()

    def _pad(self,array,pad_value=0):
        return array + [pad_value] * (self.max_len - len(array))

    def _build_team_vocab(self,vocab_file):
        """构建球队名到ID的映射（从0开始编码）"""
        team2id,id2team ={},{}

        for line in open(vocab_file,"r"):
            idx,team = line.strip("\n").split("\t")
            team2id[team] = int(idx)
            id2team[int(idx)] = team

        return team2id, id2team

    def _generate_label(self):
        """生成标签：0=负（A负B），1=平，2=胜（A胜B）"""
        df = self.raw_data.copy()

        def get_label(row):
            if row["主场得分"] > row["客场得分"]:
                return 2  # A胜
            elif row["主场得分"] == row["客场得分"]:
                return 1  # 平
            else:
                return 0  # A负

        df["label"] = df.apply(get_label, axis=1)
        return df

    def _build_sequences(self):
        """
        为每个比赛样本构建时序序列
        核心：为每支球队记录最近seq_len场的对阵信息，作为LSTM的输入序列
        """
        # 存储每支球队的历史比赛记录（按时间）
        processed_samples = []

        for team_a, sub_data in self.raw_data.groupby("主场"):
            # 修复数据泄露：按时间排序，确保序列顺序正确
            sub_data = sub_data.sort_values("时间", ascending=True)
            team_a_ids = []
            team_b_ids = []
            masks = []
            labels = []
            caculates = []  # 保存时间信息
            # 转换球队名为ID
            for idx,row in sub_data.iterrows():
                team_a_id = self.team2id[row["主场"]]
                team_b_id = self.team2id[row["客场"]]
                team_a_ids.append(team_a_id)
                team_b_ids.append(team_b_id)
                masks.append(1)
                labels.append(row["label"])

                if self.start_time<= row["时间"] < self.end_time:
                    caculates.append(1)  # 保存时间
                else:
                    caculates.append(0)
            # 保存处理后的样本
            processed_samples.append({
                "team_a_id": self._pad(team_a_ids) if self.train else team_a_ids,
                "team_b_id": self._pad(team_b_ids) if self.train else team_b_ids,
                "masks": self._pad(masks) if self.train else masks,
                "labels": self._pad(labels) if self.train else labels,
                "caculates": self._pad(caculates) if self.train else caculates,  # 保存时间序列
            })

        return processed_samples

    def __len__(self):
        """返回样本总数"""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """获取单个样本，转换为Tensor"""
        sample = self.processed_data[idx]

        # 转换为PyTorch Tensor（适配LSTM输入格式）
        team_a_id = torch.tensor(sample["team_a_id"], dtype=torch.long)
        team_b_id = torch.tensor(sample["team_b_id"], dtype=torch.long)
        masks = torch.tensor(sample["masks"], dtype=torch.long)
        # LSTM序列：(seq_len, 2) → 注意batch_first=True，后续DataLoader会加batch维度
        label = torch.tensor(sample["labels"], dtype=torch.long)
        caculates = torch.tensor(sample["caculates"], dtype=torch.long)  # 时间信息保持为列表

        return {
            "team_a_id": team_a_id,
            "team_b_id": team_b_id,
            "mask": masks,
            "label": label,
            "caculates": caculates  # 返回时间信息
        }

class MyDataset2(Dataset):
    """
    LSTM比赛分类Dataset
    输入：原始比赛数据（比赛时间，球队a，球队b，比分）
    输出：适配LSTM的样本（球队A ID、球队B ID、时序特征、标签）
    """
    def __init__(self,
                 data,  # 数据文件路径（csv格式）
                 start_time,
                 end_time,
                 vocab_file="",
                 train=True,
                ):  # 是否为训练集（区分训练/测试）
        self.train = train
        self.start_time = start_time
        self.end_time = end_time
        # 1. 加载并预处理原始数据
        self.raw_data = data[(data["时间"]>=start_time) & (data["时间"]<end_time)]
        self.max_len = max(self.raw_data.groupby("主场").count()["id"])

        # 2. 球队名映射为整数ID（适配Embedding层）
        self.team2id, self.id2team = self._build_team_vocab(vocab_file)
        self.num_teams = len(self.team2id)
        # 3. 生成标签（胜/平/负）
        self.raw_data = self._generate_label()
        # 4. 为每个比赛样本构建时序特征序列
        self.processed_data = self._build_sequences()

    def _build_team_vocab(self,vocab_file):
        """构建球队名到ID的映射（从0开始编码）"""
        team2id,id2team ={},{}

        for line in open(vocab_file,"r"):
            idx,team = line.strip("\n").split("\t")
            team2id[team] = int(idx)
            id2team[int(idx)] = team

        return team2id, id2team

    def _generate_label(self):
        """生成标签：0=负（A负B），1=平，2=胜（A胜B）"""
        df = self.raw_data.copy()
        def get_label(row):
            if row["主场得分"] > row["客场得分"]:
                return 2  # A胜
            elif row["主场得分"] == row["客场得分"]:
                return 1  # 平
            else:
                return 0  # A负

        df["label"] = df.apply(get_label, axis=1)
        return df

    def _build_sequences(self):
        """
        为每个比赛样本构建时序序列
        核心：为每支球队记录最近seq_len场的对阵信息，作为LSTM的输入序列
        """
        # 存储每支球队的历史比赛记录（按时间）
        processed_samples = []
        for idx,row in self.raw_data.sort_values("时间", ascending=True).iterrows():
            team_a_id = self.team2id[row["主场"]]
            team_b_id = self.team2id[row["客场"]]
            # 保存处理后的样本
            processed_samples.append({
                "team_a_id": team_a_id,
                "team_b_id": team_b_id,
                "diff": row["主场得分"] - row["客场得分"],
                "label": row["label"],
            })

        return processed_samples

    def __len__(self):
        """返回样本总数"""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """获取单个样本，转换为Tensor"""
        sample = self.processed_data[idx]
        return {
            "team_a_id": sample["team_a_id"],
            "team_b_id": sample["team_b_id"],
            "diff": sample["diff"],
            "label": sample["label"]
        }

# ===================== 测试代码（验证Dataset可运行） =====================
if __name__ == "__main__":
    # 第二步：初始化Dataset
    data = pd.read_csv("data/processed/competition.csv")
    data = data.sort_values("时间", ascending=True)
    dataset = MyDataset(
        data=data,
        start_time="2022-01-01", end_time="2025-09-01",
        vocab_file="data/processed/team_ids",
        train=True
    )
    print(f"样本总数：{len(dataset)}")
    print(f"球队总数：{dataset.num_teams}")

    # 第四步：用DataLoader批量加载
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,  # 训练集打乱，测试集不打乱
        num_workers=0,  # 新手建议设0，避免多线程问题
    )

    # 遍历批量数据
    for batch in dataloader:
        print("\n批量数据：")
        print(f"批量球队A ID：{batch['team_a_id']}")
        print(f"批量mask：{batch['mask']}")
        print(f"批量标签：{batch['label']}")
        print(f"批量计算：{batch['caculates']}")
        break