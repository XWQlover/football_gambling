import pandas as pd

def revenue(preds,label:pd.Series,odds:pd.Series):
    """
    假设每场投一块，预测正确则获取相应的赔率金额，求ROI
    """
    cost = 0.0
    revenue = 0.0
    for a,b,c in zip(preds,label,odds):
        if a==b:
            # print(a,b,c)
            revenue += float(c[b])
        cost +=1
    return revenue/cost

def revenue2(preds,label:pd.Series,odds:pd.Series):
    """
    假设每场投一块，预测项变为主队胜利和主队负平，当平和负的最小赔率大于2的时候才下注，下注时平负都买
    """
    cost = 0.0
    revenue = 0.0
    for a,b,c in zip(preds,label,odds):
        if a == 2:
            if a==b:
                # print(a,b,c)
                revenue += float(c[b])
            cost +=1
        elif min(float(c[0]),float(c[1])) > 2:
            if a==b:
                revenue += float(c[b])
            cost +=2
    return revenue/cost