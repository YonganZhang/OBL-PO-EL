# import joblib
# import numpy as np
# # 导入普通模块
# import warnings
# warnings.filterwarnings('ignore')
# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
# from matplotlib import rcParams
# rcParams['axes.unicode_minus'] = False  # 正常显示符号
# # 导入自己模块
# from Z0datainput import datainput
# from Z1distr import distr
# from Z2modeltrain import modeltrain
# from Z3evaluate import eva
# from lightgbm import LGBMRegressor
# from sklearn.compose import TransformedTargetRegressor
# from sklearn.dummy import DummyRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.linear_model._glm import GeneralizedLinearRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler
# import warnings
# from sklearn.linear_model import LinearRegression, BayesianRidge, RidgeCV, Lars, ElasticNetCV, \
#     OrthogonalMatchingPursuitCV, OrthogonalMatchingPursuit, LassoLarsIC, TweedieRegressor, LassoCV, LassoLars, \
#     ElasticNet, RANSACRegressor, HuberRegressor, PassiveAggressiveRegressor, LarsCV, LassoLarsCV
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR, LinearSVR, NuSVR
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge
# from sklearn.neural_network import MLPRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.tree import ExtraTreeRegressor
# from xgboost import XGBRegressor
#
# from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import BaggingRegressor
# from sklearn.linear_model import SGDRegressor
# def my_predict(k, position, y_test, scale_y, x_train, y_train, x_test, scale_x):
#     models = [LinearRegression(), SVR(), Ridge(), MLPRegressor(alpha=20), RandomForestRegressor(),
#               AdaBoostRegressor(), GradientBoostingRegressor(), BaggingRegressor(), SGDRegressor(), BayesianRidge(),
#               RidgeCV(), KernelRidge(), Lars(), TransformedTargetRegressor(), GeneralizedLinearRegressor(),
#               ElasticNetCV(),
#               TweedieRegressor(), LassoCV(), LassoLarsIC(), OrthogonalMatchingPursuit(),
#               OrthogonalMatchingPursuitCV(),
#               HuberRegressor(), LinearSVR(), ExtraTreesRegressor(), DecisionTreeRegressor(),
#               PassiveAggressiveRegressor(),
#               LGBMRegressor(), HistGradientBoostingRegressor(), RANSACRegressor(), ExtraTreeRegressor(),
#               KNeighborsRegressor(),
#               GaussianProcessRegressor(), XGBRegressor(), ElasticNet(), Lasso(), DummyRegressor(), LassoLars(),
#               NuSVR(),
#               LarsCV(), LassoLarsCV()]
#     models_str = ['LinearRegression', 'SVR', 'Ridge', 'MLPRegressor', 'RandomForest',
#                   'AdaBoost', 'GradientBoost', 'Bagging', 'SGDRegressor', 'BayesianRidge',
#                   'RidgeCV', 'KernelRidge', 'Lars', 'TransformedTargetRegressor', 'GeneralizedLinearRegressor',
#                   'ElasticNetCV',
#                   'TweedieRegressor', 'LassoCV', 'LassoLarsIC', 'OrthogonalMatchingPursuit',
#                   'OrthogonalMatchingPursuit',
#                   'HuberRegressor', 'LinearSVR', 'ExtraTreesRegressor', 'DecisionTreeRegressor',
#                   'PassiveAggressiveRegressor',
#                   'LGBMRegressor', 'HistGradientBoostingRegressor', 'RANSACRegressor', 'ExtraTreeRegressor',
#                   'KNeighborsRegressor',
#                   'GaussianProcessRegressor', 'XGBRegressor', 'ElasticNet', 'Lasso', 'DummyRegressor', 'LassoLars',
#                   'NuSVR',
#                   'LarsCV', 'LassoLarsCV'
#                   ]
#     pre1 = []
#     pre2 = []
#     pre3 = []
#     ##对测试集先进行反归一化（不能放到下面for里，否则连续归一化很多次）
#     # y_test = y_test.ravel()
#     y_test = scale_y.inverse_transform(y_test)
#     i = 0
#     count = np.zeros(shape=(len(y_test), len(y_test[0])), dtype=int)
#     Zfinal_PRE = np.zeros(shape=(len(y_test), len(y_test[0])), dtype=int)
#     Zfinal_TUR = np.zeros(shape=(len(y_test), len(y_test[0])), dtype=int)
#     for name, model in zip(models_str, models):
#         if (position[i] > 0):
#             model_each_predict = np.zeros(shape=(len(y_test), len(y_test[0])), dtype=int)
#             for j in range(k):
#                 model = joblib.load('./model./' + name + j.__str__() + '.dat')
#                 y_pred = model.predict(x_test)
#                 y_pred2 = np.array(scale_y.inverse_transform(y_pred))
#                 for i1 in range(len(y_pred2)):
#                     count = count + position[i]  # 将权重累加，最终被除
#                     if 70 > y_pred2[i1][0] > 0 and 10 > y_pred2[i1][1] > 0:
#                         model_each_predict = model_each_predict + y_pred
#                     else:
#                         print('./model./' + name + j.__str__() + '.dat' + "第" + str(i1) + "行数据错误")
#             model_each_predict = model_each_predict / k
#             ##对预测先进行反归一化
#             # y_pred=model_each_predict.ravel()
#             y_pred = scale_y.inverse_transform(model_each_predict)
#             Zfinal_PRE = np.array([y_pred]) * position[i] + Zfinal_PRE  # 累加
#             Zfinal_TUR = np.array([y_test]) * position[i] + Zfinal_TUR
#             pre1.extend(y_pred)  # 用于画泰勒图
#             pre3.extend(y_test)  # 用于画泰勒图
#             for j in range(len(y_pred)):
#                 pre2.append(name)  # 用于画泰勒图
#         i = i + 1
#     x_test11 = scale_x.inverse_transform(x_test)
#     import xlwt
#     f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
#     sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表
#     Zfinal_TUR = [x / count for x in (Zfinal_TUR.tolist())[0]]
#     Zfinal_PRE = [x / count for x in (Zfinal_PRE.tolist())[0]]
#     for i in range(len(Zfinal_TUR)):
#         sheet1.write(i, 0, str(Zfinal_TUR[i][0]))  # 写入数据参数对应 行, 列, 值
#         sheet1.write(i, 1, str(Zfinal_TUR[i][1]))
#         sheet1.write(i, 2, str(Zfinal_PRE[i][0]))
#         sheet1.write(i, 3, str(Zfinal_PRE[i][1]))  # 写入数据参数对应 行, 列, 值
#         for j in range(7):
#             sheet1.write(i, j + 4, str(x_test11[i][j]))
#
#     f.save('最终模型结果.xls')  # 保存.xls到当前工作目录
#     print("完成集成学习")