#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理脚本：处理 raw 目录中的 4 个 Excel 文件
"""
import pandas as pd
import os
import traceback

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义输入输出路径
RAW_DIR = os.path.join(SCRIPT_DIR, 'raw')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'processed')

if __name__ == "__main__":
    data = pd.read_excel(f"{RAW_DIR}/新浪彩票.xlsx",sheet_name=0)
    data = data.dropna(subset=["主场","客场","主场得分","客场得分"])
    data["时间"] = data["时间"].apply(lambda x:x[:16])
    data["id"] = data["数据来源"].apply(
                lambda x:x.replace("https://lottery.sina.com.cn/ai/match/football/base.shtml?matchId=","").replace("&sportsType=football",""))
    for col in ["胜预测值","平预测值","负预测值"]:
        data[col] = data[col].apply(lambda x:str(x).strip("%"))
    data[["id", '主场', '客场', '比赛', '时间', '主场得分', '客场得分', '胜赔率', '平赔率', '负赔率', '胜预测值','平预测值', '负预测值']].sort_values("时间",ascending=True).to_csv(f"{PROCESSED_DIR}/competition.csv",index=False)

    # 球队token化
    teams = list(set(data["主场"].tolist() + data["客场"].tolist()))
    teams.sort()
    f = open(f"{PROCESSED_DIR}/team_ids","w")
    for idx,team in enumerate(teams):
        f.write(str(idx) + '\t' + team + "\n")

    data = pd.read_excel(f"{RAW_DIR}/赛前赔率.xlsx", sheet_name=0)
    data = data.dropna(subset=["胜", "负", "平"])
    data["id"] = data["任务源网址"].apply(
        lambda x: x.replace("https://lottery.sina.com.cn/ai/match/football/spf.shtml?matchId=", "").replace(
            "&sportsType=football", ""))
    for col in ["胜概率","平概率","负概率","返概率"]:
        data[col] = data[col].apply(lambda x: str(x).strip("%"))
    data[["id","胜","平","负","胜概率","平概率","负概率","返概率","凯利走势1","凯利走势2","凯利走势3"]].to_csv(f"{PROCESSED_DIR}/odds.csv",index=False)

    data = pd.read_excel(f"{RAW_DIR}/战绩.xlsx", sheet_name=0)
    data["id"] = data["任务源网址"].apply(
        lambda x: x.replace("https://lottery.sina.com.cn/ai/match/football/jc.shtml?matchId=", "").replace("&sportsType=football", "")
    )
    for col in ["客场最近10场胜率","客场最近10场赢率","最近10场主场胜率","最近10场主场赢率","最近10场同主客主场胜率","最近10场同主客主场赢率","主场最近10场胜率","主场最近10场赢率"]:
        data[col] = data[col].apply(lambda x: str(x).strip("%"))

    data["最近10场主场胜场率"] = data["最近10场主场胜场"] / data[["最近10场主场胜场", "最近10场主场平场", "最近10场主场负场"]].sum(axis=1)
    data["最近10场主场平场率"] = data["最近10场主场平场"] / data[["最近10场主场胜场", "最近10场主场平场", "最近10场主场负场"]].sum(axis=1)
    data["最近10场主场负场率"] = data["最近10场主场负场"] / data[["最近10场主场胜场", "最近10场主场平场", "最近10场主场负场"]].sum(axis=1)

    data["最近10场同主客胜场率"] = data["最近10场同主客胜场"] / data[["最近10场同主客胜场", "最近10场同主客平场", "最近10场同主客负场"]].sum(axis=1)
    data["最近10场同主客平场率"] = data["最近10场同主客平场"] / data[["最近10场同主客胜场", "最近10场同主客平场", "最近10场同主客负场"]].sum(axis=1)
    data["最近10场同主客负场率"] = data["最近10场同主客负场"] / data[["最近10场同主客胜场", "最近10场同主客平场", "最近10场同主客负场"]].sum(axis=1)

    data["主场最近10场胜场率"] = data["主场最近10场胜场"] / data[["主场最近10场胜场", "主场最近10场平场", "主场最近10场负场"]].sum(axis=1)
    data["主场最近10场平场率"] = data["主场最近10场平场"] / data[["主场最近10场胜场", "主场最近10场平场", "主场最近10场负场"]].sum(axis=1)
    data["主场最近10场负场率"] = data["主场最近10场负场"] / data[["主场最近10场胜场", "主场最近10场平场", "主场最近10场负场"]].sum(axis=1)

    data["客场最近10场胜场率"] = data["客场最近10场胜场"] / data[["客场最近10场胜场", "客场最近10场平场", "客场最近10场负场"]].sum(axis=1)
    data["客场最近10场平场率"] = data["客场最近10场平场"] / data[["客场最近10场胜场", "客场最近10场平场", "客场最近10场负场"]].sum(axis=1)
    data["客场最近10场负场率"] = data["客场最近10场负场"] / data[["客场最近10场胜场", "客场最近10场平场", "客场最近10场负场"]].sum(axis=1)

    data[
        ["id", "最近10场主场胜场率", "最近10场主场平场率", "最近10场主场负场率", "最近10场同主客胜场率", "最近10场同主客平场率", "最近10场同主客负场率",
         "主场最近10场胜场率", "主场最近10场平场率", "主场最近10场负场率", "客场最近10场胜场率","客场最近10场平场率","客场最近10场负场率",
         "客场最近10场胜率","客场最近10场赢率","最近10场主场胜率","最近10场主场赢率","最近10场同主客主场胜率","最近10场同主客主场赢率","主场最近10场胜率","主场最近10场赢率"]].to_csv(
        f"{PROCESSED_DIR}/records.csv", index=False)