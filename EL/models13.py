import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from lazypredict.Supervised import LazyRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model._glm import GeneralizedLinearRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.linear_model import LinearRegression, BayesianRidge, RidgeCV, Lars, ElasticNetCV, \
    OrthogonalMatchingPursuitCV, OrthogonalMatchingPursuit, LassoLarsIC, TweedieRegressor, LassoCV, LassoLars, \
    ElasticNet, RANSACRegressor, HuberRegressor, PassiveAggressiveRegressor, LarsCV, LassoLarsCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import SGDRegressor

## 3.将模型和他们的名字分别放到两个列表里，然后建一个空列表保存他们的得分
from Z3evaluate import evaluate


def models15(k_num,x_train1, x_test1, y_train1, y_test1, scale_x,scale_y):
    models = [LinearRegression(), SVR(), Ridge(), MLPRegressor(alpha=20), RandomForestRegressor(),
              AdaBoostRegressor(), GradientBoostingRegressor(), BaggingRegressor(), SGDRegressor(), BayesianRidge(),
              RidgeCV(), KernelRidge(), Lars(), TransformedTargetRegressor(), GeneralizedLinearRegressor(),
              ElasticNetCV(),
              TweedieRegressor(), LassoCV(), LassoLarsIC(), OrthogonalMatchingPursuit(), OrthogonalMatchingPursuitCV(),
              HuberRegressor(), LinearSVR(), ExtraTreesRegressor(), DecisionTreeRegressor(),
              PassiveAggressiveRegressor(),
              LGBMRegressor(), HistGradientBoostingRegressor(), RANSACRegressor(), ExtraTreeRegressor(),
              KNeighborsRegressor(),
              GaussianProcessRegressor(), XGBRegressor(), ElasticNet(), Lasso(), DummyRegressor(), LassoLars(), NuSVR(),
              LarsCV(), LassoLarsCV()]
    models_str = ['LinearRegression', 'SVR', 'Ridge', 'MLPRegressor', 'RandomForest',
                  'AdaBoost', 'GradientBoost', 'Bagging', 'SGDRegressor', 'BayesianRidge',
                  'RidgeCV', 'KernelRidge', 'Lars', 'TransformedTargetRegressor','GeneralizedLinearRegressor',
                  'ElasticNetCV',
                  'TweedieRegressor', 'LassoCV', 'LassoLarsIC', 'OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuit',
                  'HuberRegressor', 'LinearSVR', 'ExtraTreesRegressor','DecisionTreeRegressor' ,
                  'PassiveAggressiveRegressor',
                  'LGBMRegressor', 'HistGradientBoostingRegressor', 'RANSACRegressor', 'ExtraTreeRegressor',
                  'KNeighborsRegressor',
                  'GaussianProcessRegressor', 'XGBRegressor', 'ElasticNet', 'Lasso', 'DummyRegressor', 'LassoLars','NuSVR',
                  'LarsCV', 'LassoLarsCV'
                  ]
    # score_1=[]
    score_2 = []
    score_all = []
    score_all_2 = []
    score_all_3 = []
    pre1 = []
    pre2 = []
    pre3 = []
    ##对测试集先进行反归一化（不能放到下面for里，否则连续归一化很多次）
    #y_test1 = y_test1.ravel()#如果预测矩阵是一维，请允许此行运行
    y_test1 = scale_y.inverse_transform(y_test1)
    for name, model in zip(models_str, models):
        model = model
        print(model)
        model = MultiOutputRegressor(model)
        model.fit(x_train1, y_train1)
        #y_train_pred = model.predict(x_train1)
        y_pred = model.predict(x_test1)
        ##对预测先进行反归一化
        #y_pred=y_pred.ravel()#如果预测的是一维数组，请加上这句，将原本转为二维的换成一维回来
        y_pred=scale_y.inverse_transform(y_pred)
        joblib.dump(model, "./model/" + name + k_num.__str__() + '.dat')
        # score_2 = evaluate(y_pred, y_test1)
        # score_1 = evaluate(y_train_pred, y_train1)
        # score_3 = model.score(x_train1, y_train1)
        # score_all_1.append(score_1)
        # score_all.append(score_2)
        # score_all_3.append(score_3)
        # print(name + '得分:' + str(score_1))
        # score_3 = model.score(x_train1, y_train1)#3是训练准确率的R2值
        # score_all_3.append(score_3)
        # score_2 = evaluate(y_pred, y_test1)
        # score_all_2.append(score_2)
        pre1.extend(y_pred)#用于画泰勒图
        pre3.extend(y_test1)  # 用于画泰勒图
        for i in range(len(y_pred)):
            pre2.append(name)  # 用于画泰勒图
    # return score_all_3, score_all_2 #3为训练集的R2,2为测试集的全部指标


    return [[pre2],[pre3],[pre1]],models_str#模型名字、真值、预测值

