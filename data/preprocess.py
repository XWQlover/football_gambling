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
GAME1 = "传统"
GAME2 = "竞彩"
# 定义输入输出路径
RAW_DIR = os.path.join(SCRIPT_DIR, 'raw')
PROCESSED_DIR = os.path.join(SCRIPT_DIR, 'processed')

if __name__ == "__main__":
    data1 = pd.read_excel(f"{RAW_DIR}/{GAME1}/比分.xlsx",sheet_name=0)
    data2 = pd.read_excel(f"{RAW_DIR}/{GAME2}/比分.xlsx", sheet_name=0)
    data = pd.concat([data1,data2])
    data = data.dropna(subset=["主队","客队","主队得分","客队得分"])
    data["比赛时间"] = data["比赛时间"].apply(lambda x:x[:16])
    data["id"] = data["页面网址"].apply(
                lambda x:x.replace("https://lottery.sina.com.cn/ai/match/football/base.shtml?matchId=","").replace("&sportsType=football",""))
    for col in ["主队预测胜率","平预测胜率","客队预测胜率"]:
        data[col] = data[col].apply(lambda x:str(x).strip("%"))
    data.sort_values("比赛时间",ascending=True).to_csv(f"{PROCESSED_DIR}/competition.csv",index=False)

    # 球队token化
    teams = list(set(data["主队"].tolist() + data["客队"].tolist()))
    teams.sort()
    f = open(f"{PROCESSED_DIR}/team_ids","w")
    for idx,team in enumerate(teams):
        f.write(str(idx) + '\t' + team + "\n")
    #
    # data = pd.read_excel(f"{RAW_DIR}/赛前赔率.xlsx", sheet_name=0)
    # data = data.dropna(subset=["胜", "负", "平"])
    # data["id"] = data["任务源网址"].apply(
    #     lambda x: x.replace("https://lottery.sina.com.cn/ai/match/football/spf.shtml?matchId=", "").replace(
    #         "&sportsType=football", ""))
    # for col in ["胜概率","平概率","负概率","返概率"]:
    #     data[col] = data[col].apply(lambda x: str(x).strip("%"))
    # data[["id","胜","平","负","胜概率","平概率","负概率","返概率","凯利走势1","凯利走势2","凯利走势3"]].to_csv(f"{PROCESSED_DIR}/odds.csv",index=False)
    #
    # data = pd.read_excel(f"{RAW_DIR}/战绩.xlsx", sheet_name=0)
    # data["id"] = data["任务源网址"].apply(
    #     lambda x: x.replace("https://lottery.sina.com.cn/ai/match/football/jc.shtml?matchId=", "").replace("&sportsType=football", "")
    # )
    # for col in ["客场最近10场胜率","客场最近10场赢率","最近10场主场胜率","最近10场主场赢率","最近10场同主客主场胜率","最近10场同主客主场赢率","主场最近10场胜率","主场最近10场赢率"]:
    #     data[col] = data[col].apply(lambda x: str(x).strip("%"))
    #
    # data["最近10场主场胜场率"] = data["最近10场主场胜场"] / data[["最近10场主场胜场", "最近10场主场平场", "最近10场主场负场"]].sum(axis=1)
    # data["最近10场主场平场率"] = data["最近10场主场平场"] / data[["最近10场主场胜场", "最近10场主场平场", "最近10场主场负场"]].sum(axis=1)
    # data["最近10场主场负场率"] = data["最近10场主场负场"] / data[["最近10场主场胜场", "最近10场主场平场", "最近10场主场负场"]].sum(axis=1)
    #
    # data["最近10场同主客胜场率"] = data["最近10场同主客胜场"] / data[["最近10场同主客胜场", "最近10场同主客平场", "最近10场同主客负场"]].sum(axis=1)
    # data["最近10场同主客平场率"] = data["最近10场同主客平场"] / data[["最近10场同主客胜场", "最近10场同主客平场", "最近10场同主客负场"]].sum(axis=1)
    # data["最近10场同主客负场率"] = data["最近10场同主客负场"] / data[["最近10场同主客胜场", "最近10场同主客平场", "最近10场同主客负场"]].sum(axis=1)
    #
    # data["主场最近10场胜场率"] = data["主场最近10场胜场"] / data[["主场最近10场胜场", "主场最近10场平场", "主场最近10场负场"]].sum(axis=1)
    # data["主场最近10场平场率"] = data["主场最近10场平场"] / data[["主场最近10场胜场", "主场最近10场平场", "主场最近10场负场"]].sum(axis=1)
    # data["主场最近10场负场率"] = data["主场最近10场负场"] / data[["主场最近10场胜场", "主场最近10场平场", "主场最近10场负场"]].sum(axis=1)
    #
    # data["客场最近10场胜场率"] = data["客场最近10场胜场"] / data[["客场最近10场胜场", "客场最近10场平场", "客场最近10场负场"]].sum(axis=1)
    # data["客场最近10场平场率"] = data["客场最近10场平场"] / data[["客场最近10场胜场", "客场最近10场平场", "客场最近10场负场"]].sum(axis=1)
    # data["客场最近10场负场率"] = data["客场最近10场负场"] / data[["客场最近10场胜场", "客场最近10场平场", "客场最近10场负场"]].sum(axis=1)
    #
    # data[
    #     ["id", "最近10场主场胜场率", "最近10场主场平场率", "最近10场主场负场率", "最近10场同主客胜场率", "最近10场同主客平场率", "最近10场同主客负场率",
    #      "主场最近10场胜场率", "主场最近10场平场率", "主场最近10场负场率", "客场最近10场胜场率","客场最近10场平场率","客场最近10场负场率",
    #      "客场最近10场胜率","客场最近10场赢率","最近10场主场胜率","最近10场主场赢率","最近10场同主客主场胜率","最近10场同主客主场赢率","主场最近10场胜率","主场最近10场赢率"]].to_csv(
    #     f"{PROCESSED_DIR}/records.csv", index=False)