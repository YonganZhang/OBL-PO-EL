import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as SS

def datainput():
    data = pd.read_csv('data.csv', encoding='utf-8',header=None)
    ## 2.数据预处理
    # Id=data.loc[:,'Id']   #ID先提取出来，后面合并表格要用
    # data=data.drop('Id',axis=1)
    x = data.loc[:, 0:15]
    y = data.loc[:, 16:17]

    # mean_cols=x.mean()
    # x=x.fillna(mean_cols)  #填充缺失值
    # x_dum=pd.get_dummies(x)    #独热编码

    # 再整理出一组标准化的数据，通过对比可以看出模型的效果有没有提高
    from sklearn.preprocessing import MinMaxScaler
    # scaler_x = MinMaxScaler()
    # scaler_x.fit(x)
    # scaler_y = MinMaxScaler()
    # y=np.array(y).reshape(-1,1)
    # scaler_y.fit(y)
    # x1 = scaler_x.transform(x)
    # y1 = scaler_y.transform(y)
    zzz = []
    # 建立标准化器
    scale_x = SS()
    scale_y = SS()
    y1=np.array(y).tolist()
    if isinstance(y1[0], list):
        y1 =np.array(y1)
    else:
        # 变成列二维list
        y1=np.array(y1).reshape(-1, 1)
    # 标准化
    x1 = scale_x.fit_transform(x)
    y1 = scale_y.fit_transform(y1)
    # 建立标准化器
    #y1 = y1.ravel()，如果是预测矩阵一维，请允许运行此行
    return x1,y1,scale_x,scale_y