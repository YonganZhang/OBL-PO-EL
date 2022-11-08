import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model._glm import GeneralizedLinearRegressor
from sklearn.model_selection import train_test_split
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
import xlwt

from Z3evaluate import evaluate


def other_models(K,y_test,scale_y,scale_x,x_train, y_train,x_test):
    num_class=7#有num_class个层类
    error_type=3#3就是R2

    f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表
    sheet2 = f.add_sheet('sheet2', cell_overwrite_ok=True)  # 创建sheet工作表
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
    ##对测试集先进行反归一化（不能放到下面for里，否则连续归一化很多次）
    #y_test = y_test.ravel()
    y_test = scale_y.inverse_transform(y_test)
    # for name, model in zip(models_str, models):
    #     model = model
    #     model.fit(x_train, y_train)
    #     y_pred = model.predict(x_test)
    #     ##对预测先进行反归一化
    #     y_pred=y_pred.ravel()
    #     y_pred=scale_y.inverse_transform(y_pred)
    #     Zfinal_PRE=np.array([y_pred])
    #     Zfinal_TUR=np.array([y_test])
    #     Zfinal_TUR = (Zfinal_TUR.tolist())[0]
    #     Zfinal_PRE = (Zfinal_PRE.tolist())[0]
    #     for i in range(len(Zfinal_TUR)):
    #         sheet1.write(count, 1, name)  # 写入数据参数对应 行, 列, 值
    #         sheet1.write(count, 2, str(Zfinal_TUR[i]))  # 写入数据参数对应 行, 列, 值
    #         sheet1.write(count, 3, str(Zfinal_PRE[i]))  # 写入数据参数对应 行, 列, 值
    #         count = count + 1
    #f.save('其他模型结果.xls')  # 保存.xls到当前工作目录
    TT2=-1#记录模型的第几个个数，用来保存各个模型的评分结果
    count = 0#记录excel保存预测结果的行数
    for name, model in zip(models_str, models):
        class_level_TUR = [[]  for _ in range(num_class)]  # 记录各个层的真实结果，每个模型不一样因此每次循环置为空
        class_level_PRE = [[]  for _ in range(num_class)]  # 记录各个层的分类结果
        TT2 = TT2 + 1
        eva2 = [[[]  for _ in range(2)]  for _ in range(num_class)]
        model_each_predict = np.zeros(shape = (len(y_test), len(y_test[0])),dtype=int)
        for j in range(K):
            model = joblib.load('./model./' + name + j.__str__() + '.dat')
            y_pred = model.predict(x_test)
            model_each_predict = model_each_predict + y_pred
        model_each_predict = model_each_predict / K
        ##对预测先进行反归一化
        #y_pred=model_each_predict.ravel()
        y_pred=scale_y.inverse_transform(model_each_predict)
        Zfinal_PRE=np.array([y_pred])
        Zfinal_TUR=np.array([y_test])
        Zfinal_TUR = (Zfinal_TUR.tolist())[0]
        Zfinal_PRE = (Zfinal_PRE.tolist())[0]
        ##存入其他模型的预测值
        for i in range(len(Zfinal_TUR)):
            sheet1.write(count, 0, name)  # 写入数据参数对应 行, 列, 值
            TT=1
            for j in range(len(Zfinal_TUR[0])):
                sheet1.write(count, TT, str(Zfinal_TUR[i][j]))
                sheet1.write(count, TT+1, str(Zfinal_PRE[i][j]))
                TT=TT+2
        ##存入其他模型的预测层号
            x_test11 = scale_x.inverse_transform(x_test)
            for j in range(num_class):
                 sheet1.write(count, j + 5, str(x_test11[count%len(x_test)][j]))
            count = count + 1#此行不能删，不然大乱

        ##存入各个层号的评估分数
            for j in range(num_class):
                if (abs(x_test11[i][j]-1))<0.005:
                    class_level_TUR[j].append(Zfinal_TUR[i])
                    class_level_PRE[j].append(Zfinal_PRE[i])
        for j in range(num_class):
            eva2[j][0] = evaluate(class_level_PRE[j][0], class_level_TUR[j][0])[error_type]
            eva2[j][1] = evaluate(class_level_PRE[j][1], class_level_TUR[j][1])[error_type]
        ##存入其他模型的评估
        eva=evaluate(Zfinal_PRE,Zfinal_TUR)
        sheet2.write(TT2, 0, str(name))
        # sheet2.write(TT2, 1 + 1, str(eva))
        for j in range(len(eva)):
            sheet2.write(TT2, 2+j, str(eva[j]))#j+1是因为第一行要保存模型名字
        for j2 in range(num_class):
            for j3 in range(2):
                sheet2.write(TT2, 5+len(eva)+j2*3+j3, str(eva2[j2][j3]))#j+1是因为第一行要保存模型名字
    f.save('其他模型结果.xls')  # 保存.xls到当前工作目录