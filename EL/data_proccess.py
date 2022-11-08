import numpy as np
import pandas as pd


def getpos(n):
    now=n
    excel = pd.read_excel('./最新权重.xlsx',header=None)
    iloc_ = excel.iloc[now:now+1, :]
    iloc_=np.array(iloc_)
    return iloc_
