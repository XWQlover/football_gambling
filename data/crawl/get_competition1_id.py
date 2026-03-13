import requests
import time
import re

def extract_match_ids(text):
    # 正则表达式：匹配 matchId= 后面的数字
    pattern = r'matchId=(\d+)'
    # 找到所有匹配的数字
    matches = re.findall(pattern, text)
    return set(matches)

initial = 22001
start = initial
end = 26033
fr = open("传统/传统足彩ids.txt", "r")
fw = open("传统/传统足彩ids.txt", "a+")
if __name__ == "__main__":
    competitions = set()
    for line in fr.readlines():
        items_num, macth_id = line.strip("\n").split("\t")
        competitions.add(int(items_num))

    fail = 0
    while start <= end:
        if start in competitions:
            start += 1
            continue
        print(start)
        url = f"https://view.lottery.sina.com.cn/lottery_index/sfc/index?num={start}&dpc=1"
        res = requests.get(url).text
        if "match" in res:
            match_ids=extract_match_ids(res)
            for matchid in match_ids:
                fw.write(f"{start}\t{matchid}\n")
            fail = 0
        else:
            fail += 1
        if fail == 5:
            start = initial + 1000
            initial += 1000
            fail = 0
            continue
        time.sleep(1)
        start +=1
        fw.flush()
