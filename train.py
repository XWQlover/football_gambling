#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM模型训练和测试脚本
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from model.lstm import LSTMClassifier, device
from dataset.dataset import MyDataset
import pandas as pd
from config import *
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 配置参数
config = {
    'data_path': 'data/processed/competition.csv',
    'vocab_file': 'data/processed/team_ids',
    'batch_size': 1,
    'hidden_dim': 128,
    'team_emb_dim': 64,
    'num_layers': 1,
    'dropout': 0.0,
    'learning_rate': 0.001,
    'num_epochs': 20,
    'save_dir': 'checkpoints',
    'checkpoint_path':'checkpoints/best_model.pth'
}
def train_epoch(model, dataloader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        team_a_id = batch['team_a_id'].to(device)
        team_b_id = batch['team_b_id'].to(device)
        labels = batch['label'].to(device)
        masks = batch["mask"].to(device)  # 用于mask padding位置
        caculates = batch["caculates"].to(device)
        # 前向传播
        outputs = model(team_a_id, team_b_id)  # (batch_size, seq_len, 3)
        
        # 计算损失（只对有效位置计算）
        batch_size, seq_len, num_classes = outputs.shape
        outputs_flat = outputs.view(-1, num_classes)  # (batch_size * seq_len, 3)
        labels_flat = labels.view(-1)  # (batch_size * seq_len)
        masks_flat = masks.view(-1)  # (batch_size * seq_len)
        caculates_flat = caculates.view(-1)
        # 只对mask=1的位置计算损失
        valid_indices = masks_flat == 1
        if valid_indices.sum() > 0:
            loss = criterion(outputs_flat[valid_indices], labels_flat[valid_indices])
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            caculates_indices = caculates_flat==1
            # 收集预测结果（只对有效位置）
            predictions = torch.argmax(outputs_flat[caculates_indices], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_flat[caculates_indices].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions) if len(all_labels) > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            team_a_id = batch['team_a_id'].to(device)
            team_b_id = batch['team_b_id'].to(device)
            labels = batch['label'].to(device)
            masks = batch["mask"].to(device)
            caculates = batch["caculates"].to(device)
            # 前向传播
            outputs = model(team_a_id, team_b_id)  # (batch_size, seq_len, 3)
            
            # 计算损失（只对有效位置计算）
            batch_size, seq_len, num_classes = outputs.shape
            outputs_flat = outputs.view(-1, num_classes)
            labels_flat = labels.view(-1)
            masks_flat = masks.view(-1)
            caculates_flat = caculates.view(-1)
            # 只对mask=1的位置计算损失
            valid_indices = masks_flat == 1
            if valid_indices.sum() > 0:
                loss = criterion(outputs_flat[valid_indices], labels_flat[valid_indices])
                total_loss += loss.item()
                
                # 收集预测结果
                caculates_indices = caculates_flat == 1
                # 收集预测结果（只对有效位置）
                predictions = torch.argmax(outputs_flat[caculates_indices], dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_flat[caculates_indices].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions) if len(all_labels) > 0 else 0

    return avg_loss, accuracy, all_predictions, all_labels

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
    train_dataset = MyDataset(
        data=train_data,
        vocab_file=config['vocab_file'],
        start_time="2022-01-01",end_time=val_start_time,
        train=True
    )
    # 验证集：只使用验证时间段之前的数据（包含训练数据，但不包含测试数据）
    val_data = data[data["时间"] < val_end_time].copy()
    print(f"验证集数据范围: < {val_end_time}, 样本数: {len(val_data)}")
    val_dataset = MyDataset(
        data=val_data,
        vocab_file=config['vocab_file'],
        start_time=val_start_time, end_time=val_end_time,
        train=True
    )
    # 3. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    # 4. 初始化模型
    print("=" * 50)
    print("初始化模型...")
    model = LSTMClassifier(
        hidden_dim=config['hidden_dim'],
        num_teams=train_dataset.num_teams,
        team_emb_dim=config['team_emb_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    # 6. 训练循环
    print("=" * 50)
    print("开始训练...")
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.4f})")
    
    print("=" * 50)
    print("训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.4f}")


def test():
    """测试主函数"""
    # 配置参数（需要与训练时保持一致）
    data = pd.read_csv(config['data_path'])
    data = data.sort_values("时间", ascending=True)
    # 1. 加载数据集（修复数据泄露：只传入测试时间段之前的数据）
    print("=" * 50)
    print("加载测试数据集...")
    # 测试集：只使用测试时间段之前的数据（包含训练和验证数据，但不包含测试之后的数据）
    test_data = data[data["时间"] < test_end_time].copy()
    print(f"测试集数据范围: < {test_end_time}, 样本数: {len(test_data)}")
    test_dataset = MyDataset(
        data=test_data,
        vocab_file=config['vocab_file'],
        start_time=test_start_time, end_time=test_end_time,
        train=True  # 测试时不padding
    )
    # 2. 创建DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )
    # 3. 加载模型
    print("=" * 50)
    print("加载模型...")
    model = LSTMClassifier(
        hidden_dim=config['hidden_dim'],
        num_teams=test_dataset.num_teams,
        team_emb_dim=config['team_emb_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型成功 (Epoch {checkpoint['epoch']}, 验证准确率: {checkpoint['val_acc']:.4f})")
    
    # 4. 测试
    print("=" * 50)
    print("开始测试...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion)
    test_datas = []
    for team_a, sub_data in data.groupby("主场"):
        # 修复数据泄露：按时间排序，确保序列顺序正确
        sub_data = sub_data.sort_values("时间", ascending=True)
        sub_data = sub_data[(sub_data["时间"]>=test_start_time) & (sub_data["时间"]<test_end_time)]
        test_datas.append(sub_data)

    test_datas = pd.concat(test_datas,axis=0)
    test_datas["pred"] = predictions
    test_datas.to_csv("data/eval/test_datas.csv",index=False)
    print(f"\n测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    
    # 5. 详细分类报告
    print("\n" + "=" * 50)
    print("分类报告:")
    print(classification_report(
        labels, predictions,
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

