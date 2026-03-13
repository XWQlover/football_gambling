import requests
import time
import re
from datetime import datetime, timedelta

start_time = "2024-02-18"
end_time = "2026-03-06"

start = datetime.strptime(start_time, "%Y-%m-%d")
end = datetime.strptime(end_time, "%Y-%m-%d")

current_date = start

fr = open("竞彩/竞彩ids.txt", "r")
fw = open("竞彩/竞彩ids.txt", "a+")

competitions = set()
for line in fr.readlines():
    date, macth_id = line.strip("\n").split("\t")
    competitions.add(date)

while current_date <= end:
    date = current_date.strftime('%Y-%m-%d')
    if date in competitions:
        current_date += timedelta(days=1)
        continue
    print(date)
    url = f"https://alpha.lottery.sina.com.cn/gateway/index/entry?format=json&__caller__=web&__version__=1.0.0&__verno__=1&cat1=jczqMatches&gameTypes=spf&date={date}&isPrized=&dpc=1"

    res = requests.get(url).json()

    res = res["result"]["data"]

    for e in res:
        fw.write(f"{date}\t{e['matchId']}\n")

    current_date += timedelta(days=1)
    time.sleep(1)
    fw.flush()