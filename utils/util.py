import pandas as pd

def revenue2(preds,label:pd.Series,odds:pd.Series):
    cost = 0
    revenue = 0
    for a,b,c in zip(preds,label,odds):
        print(a)
        print(b)
        print(c)
        input()