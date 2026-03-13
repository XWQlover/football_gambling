fr = open("传统足彩ids.txt", "r")
fbase = open("base_url.txt", "w")
fjc = open("jc_url.txt", "w")
fspf = open("spf_url.txt", "w")
fzb = open("zb_url.txt", "w")

if __name__ == "__main__":

    for line in fr.readlines():
        _, macth_id = line.strip("\n").split("\t")

        url = f"https://lottery.sina.com.cn/ai/match/football/base.shtml?matchId={macth_id }&sportsType=football"
        fbase.write(url + "\n")

        url = f"https://lottery.sina.com.cn/ai/match/football/jc.shtml?matchId={macth_id }&sportsType=football"
        fjc.write(url + "\n")

        url = f"https://lottery.sina.com.cn/ai/match/football/spf.shtml?matchId={macth_id }&sportsType=football"
        fspf.write(url + "\n")

        url = f"https://lottery.sina.com.cn/ai/match/football/zb.shtml?matchId={macth_id }&sportsType=football"
        fzb.write(url + "\n")