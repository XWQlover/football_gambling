import pandas as pd
import numpy as np
from config import *
from dataset.dataset import MyDataset2
from sklearn.metrics import accuracy_score, classification_report
import pickle
from tqdm import tqdm
import math
import trueskill
from tqdm import tqdm
import os
import lightgbm as lgb
from utils.util import revenue2

# 配置参数
config = {
    'competition_path': 'data/processed/competition.csv',
    'records_path': 'data/processed/records.csv',
    'odds_path': 'data/processed/odds.csv',
    'vocab_file': 'data/processed/team_ids',
    'save_dir': 'checkpoints',
    'checkpoint_path': 'checkpoints/'
}
data = pd.read_csv(config['competition_path'])
data1 = pd.read_csv(config['records_path'])
data2 = pd.read_csv(config['odds_path'])
data["label"] = data.apply(lambda x:1 if x["主场得分"] > x["客场得分"] else 0,axis=1)
# data2["赔率押注"] = data2.apply(lambda x:2 if x["胜"] == min(x[["胜","负","平"]]) else ( 0 if x["负"] == min(x[["胜","负","平"]]) else 1),axis=1)
data2["赔率押注"] = data2.apply(lambda x:1 if x["胜"]  == min(x[["胜","负","平"]]) else 0,axis=1)
# data2["胜赔率"] = data2.apply(lambda x:x["胜"]/sum(x[["胜","负","平"]]),axis=1)
# data2["平赔率"] = data2.apply(lambda x:x["平"]/sum(x[["胜","负","平"]]),axis=1)
# data2["负赔率"] = data2.apply(lambda x:x["负"]/sum(x[["胜","负","平"]]),axis=1)

data2 = pd.merge(data1,data2,left_on="id",right_on="id",how="left")
data = pd.merge(data2,data[["id","时间","label",'胜赔率','平赔率','负赔率','胜预测值','平预测值','负预测值']],left_on="id",right_on="id",how="left")
data = data.sort_values("时间", ascending=True)
data["概率押注"] = data.apply(lambda x:1 if x["胜概率"]  == min(x[["胜概率","负概率","平概率"]]) else 0,axis=1)
data["预测押注"] = data.apply(lambda x:1 if x["胜预测值"]  == min(x[["胜预测值","负预测值","平预测值"]]) else 0,axis=1)

features = ['最近10场主场胜场率', '最近10场主场平场率', '最近10场主场负场率', '最近10场同主客胜场率',
       '最近10场同主客平场率', '最近10场同主客负场率', '主场最近10场胜场率', '主场最近10场平场率', '主场最近10场负场率',
       '客场最近10场胜场率', '客场最近10场平场率', '客场最近10场负场率', '客场最近10场胜率', '客场最近10场赢率',
       '最近10场主场胜率', '最近10场主场赢率', '最近10场同主客主场胜率', '最近10场同主客主场赢率', '主场最近10场胜率',
       '主场最近10场赢率', '胜', '平', '负', '胜概率', '平概率', '负概率','胜预测值','平预测值','负预测值', '返概率', '凯利走势1',
       '凯利走势2', '凯利走势3',"赔率押注","预测押注","概率押注"]
categorical_features = ["赔率押注","预测押注","概率押注"]

params = {
    'objective': 'binary',  # 核心修改：二分类→多分类（3分类属于多分类）
    # 'num_class': 2,             # 新增：指定分类数量为3（必须加，否则模型不知道分几类）
    'metric': ['auc'],  # 核心修改：评估指标改为多分类错误率（1 - 准确率）
    'boosting_type': 'gbdt',    # 保持不变：梯度提升树类型
    'learning_rate': 0.01,      # 保持不变：学习率
    'num_leaves': 31,           # 保持不变：叶子节点数量（3分类可适当调大，比如40，视数据而定）
    'max_depth': -1,            # 保持不变：树的最大深度
    'verbose': 1,               # 保持不变：日志输出级别
    'random_state': 42,         # 保持不变：随机种子
    'feature_fraction': 0.9,    # 保持不变：特征采样比例
    "min_data_in_leaf": 6,      # 保持不变：叶子节点最小样本数
    'reg_lambda': 0.1,          # 保持不变：L2正则
    'reg_alpha': 0,             # 保持不变：L1正则
    'n_estimators': 500,        # 保持不变：树的数量
    # 可选新增：多分类场景下的优化参数
    'boost_from_average': True, # 多分类建议开启，提升稳定性
    'is_unbalance': False       # 如果3类样本不均衡，设为True；均衡则False
}

def train():
    """训练主函数"""
    # 创建保存目录
    # 1. 加载数据集（修复数据泄露：只传入对应时间段之前的数据）
    print("=" * 50)
    print("加载数据集...")
    # 训练集：只使用训练时间段之前的数据
    train_data = data[("2022-01-01"<=data["时间"]) & (data["时间"]<val_start_time)]
    print(f"训练集数据范围: < {val_start_time}, 样本数: {len(train_data)}")
    train_dataset = lgb.Dataset(train_data[features],
                             label=train_data["label"],
                             # weight=[1 + 0.0001*idx for idx,_ in enumerate(train_data.iterrows())],
                             feature_name=features,
                             categorical_feature=categorical_features)
    # 验证集：只使用验证时间段之前的数据（包含训练数据，但不包含测试数据）
    val_data = data[(data["时间"]>=val_start_time) & (data["时间"] < val_end_time)].copy()
    print(f"验证集数据范围: < {val_end_time}, 样本数: {len(val_data)}")
    val_dataset = lgb.Dataset(val_data[features],
                             label=val_data["label"],
                             feature_name=features,
                             categorical_feature=categorical_features)

    print("=" * 50)
    print("开始训练...")
    evals_result = {}
    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[val_dataset],  # 验证集
        evals_result=evals_result
    )

    # ---------------------- 5. 加载保存的模型 ----------------------
    # 验证集回炉重训
    # best_iteration = np.argmin(evals_result['valid_0']['multi_error'][100:]) + 100
    best_iteration = np.argmax(evals_result['valid_0']['auc'][100:]) + 100
    # best_score = 1 - np.min(evals_result['valid_0']['multi_error'][100:])
    best_score = np.max(evals_result['valid_0']['auc'][100:])
    best_iteration += int(best_iteration / len(train_data) * len(val_data)/2)
    params["n_estimators"] = best_iteration

    all_data = pd.concat([train_data[features], val_data[features]])
    all_dataset = lgb.Dataset(
                             all_data,
                             # weight=[1 + 0.0001 * idx for idx, _ in enumerate(all_data.iterrows())],
                             label=pd.concat([train_data["label"], val_data["label"]]),
                             feature_name=features,
                             categorical_feature=categorical_features)
    model = lgb.train(
        params,
        all_dataset,
        num_boost_round=best_iteration,
        feature_name=features
    )
    model.save_model(f"{config['save_dir']}/gbm.lgb")
    print("=" * 50)
    print("训练完成！")
    print(f"最佳验证准确率: {best_score:.4f}")
    val_acc = accuracy_score(val_data["赔率押注"], val_data["label"])
    print(f"赔率押注准确率: {val_acc:.4f}")


def test():
    """测试主函数"""
    # 1. 加载数据集（修复数据泄露：只传入测试时间段之前的数据）
    print("=" * 50)
    print("加载测试数据集...")
    test_data = data[(data["时间"]>=test_start_time) & (data["时间"] < test_end_time)].copy()
    print(f"验证集数据范围: < {val_end_time}, 样本数: {len(test_data)}")
    test_dataset = lgb.Dataset(test_data[features],
                             label=test_data["label"],
                             feature_name=features,
                             categorical_feature=categorical_features)
    # 3. 加载模型
    model = lgb.Booster(model_file=f"{config['save_dir']}/gbm.lgb")
    preds,labels = model.predict(test_data[features]),test_data["label"]
    print(preds)
    # preds = np.argmax(preds,axis=1)
    preds = np.where(preds<=0.5,0,1 )
    # 5. 详细分类报告
    print("\n" + "=" * 50)
    print("分类报告:")
    test_acc = accuracy_score(preds, labels)
    print(f"最佳验证准确率: {test_acc:.4f}")
    print(classification_report(
        labels, preds,
        target_names=['负平', '胜'],
        digits=4
    ))
    test_acc = accuracy_score(test_data["赔率押注"], labels)
    print(f"赔率押注准确率: {test_acc:.4f}")
    print(classification_report(
        labels, test_data["赔率押注"],
        target_names=['负平', '胜'],
        digits=4
    ))
    revenue2(preds,labels,test_data[['胜赔率','平赔率','负赔率']])

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
