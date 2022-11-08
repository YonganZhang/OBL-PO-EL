import random
import numpy


# https://www.jb51.net/article/241610.htm
# http://www.codeforest.cn/article/383
import numpy as np


def GWO(lb, ub, dim, SearchAgents_no, Max_iter,models_score,models_score2):
    # 初始化 alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)  # 位置.形成30的列表
    Alpha_score = float("inf")  # 这个是表示“正负无穷”,所有数都比 +inf 小；正无穷：float("inf"); 负无穷：float("-inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")  # float() 函数用于将整数和字符串转换成浮点数。

    # list列表类型
    if not isinstance(lb, list):  # 作用：来判断一个对象是否是一个已知的类型。 其第一个参数（object）为对象，第二个参数（type）为类型名，若对象的类型与参数二的类型相同则返回True
        lb = [lb] * dim  # 生成[100，100，.....100]30个
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents随机初始化狼的位置
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):  # 形成5*30个数[-100，100)以内
        Positions[:, i] = numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[
            i]  # 形成[5个0-1的数]*100-（-100）-100
    Positions[0,:]=1#保持有个属性中全部模型在线
    Convergence_curve = numpy.zeros(Max_iter)

    # 迭代寻优
    for l in range(0, Max_iter):  # 迭代1000
        for i in range(0, SearchAgents_no):  # 5
            Positions[0, :] = 1  # 保持有个属性中全部模型在线
            # 返回超出搜索空间边界的搜索代理
            # 计算每个搜索代理的目标函数（注意，这里每次代入的i指的就是狼序号，第i行参数值都是狼的属性值）
            Positions=np.array(Positions)
            fitness = F1(Positions[i, :],models_score)  # 把某行数据带入函数计算
            # print("经过计算得到：",fitness)

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        # 以上的循环里，Alpha、Beta、Delta

        a = 2 - l * ((2) / Max_iter);  # a从2线性减少到0

        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1 = random.random()  # r1 is a random number in [0,1]主要生成一个0-1的随机浮点数。
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a;  # Equation (3.3)
                C1 = 2 * r2;  # Equation (3.4)
                # D_alpha表示候选狼与Alpha狼的距离
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[
                    i, j]);  # abs() 函数返回数字的绝对值。Alpha_pos[j]表示Alpha位置，Positions[i,j])候选灰狼所在位置
                X1 = Alpha_pos[j] - A1 * D_alpha;  # X1表示根据alpha得出的下一代灰狼位置向量

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a;  #
                C2 = 2 * r2;

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j]);
                X2 = Beta_pos[j] - A2 * D_beta;

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j]);
                X3 = Delta_pos[j] - A3 * D_delta;

                Positions[i, j] = (X1 + X2 + X3) / 3  # 候选狼的位置更新为根据Alpha、Beta、Delta得出的下一代灰狼地址。

        Convergence_curve[l] = Alpha_score;

        if (l % 1 == 0):
            score_avarage=test_score_print(Alpha_pos,models_score2)
            print(['迭代次数为' + str(l) + ' 的迭代结果' + str(-Alpha_score)+' 测试集R2分数'+str(score_avarage)]);  # 每一次的迭代结果
    models_print(Alpha_pos)


def models_print(xx):#打印最终A狼的模型集
    len1 = len(xx)
    num = 0
    models_str = ['LinearRegression', 'SVR', 'Ridge', 'MLPRegressor', 'RandomForest',
                  'AdaBoost', 'GradientBoost', 'Bagging', 'SGDRegressor', 'BayesianRidge',
                  'RidgeCV', 'KernelRidge', 'Lars', 'TransformedTargetRegressor', 'GeneralizedLinearRegressor',
                  'ElasticNetCV',
                  'TweedieRegressor', 'LassoCV', 'LassoLarsIC', 'DecisionTreeRegressor', 'OrthogonalMatchingPursuit',
                  'HuberRegressor', 'LinearSVR', 'ExtraTreesRegressor', 'DecisionTreeRegressor',
                  'PassiveAggressiveRegressor',
                  'LGBMRegressor', 'HistGradientBoostingRegressor', 'RANSACRegressor', 'ExtraTreeRegressor',
                  'KNeighborsRegressor',
                  'GaussianProcessRegressor', 'XGBRegressor', 'ElasticNet', 'Lasso', 'DummyRegressor', 'LassoLars',
                  'NuSVR',
                  'LarsCV', 'LassoLarsCV'
                  ]
    print("模型共有：",end="")
    score=0
    for i in range(0, len1):
        # 如果狼的这一属性离1（有）更近，则选择改基学习器
        x1 = abs(xx[i] - 1)
        x2 = abs(xx[i] - 0)
        if x1 < x2:
            print(" "+models_str[i]+" ", end="")
            num=num+1
            score=score
    print("模型数量："+str(num))

def test_score_print(xx,models_score2):#返回每个A狼的测试分数
    len1=len(xx)
    score=0
    num=0
    for i in range(0, len1):
        # 如果狼的这一属性离1（有）更近，则选择统计该选择器的准确率
        x1 = abs(xx[i] - 1)
        x2 = abs(xx[i] - 0)
        if x1 < x2:
            score = score+models_score2[i,3]
            num = num + 1
    score_avarage=score/num
    return score_avarage

# 函数
def F1(xx,score_all):#x是狼，狼有dim个属性，每个属性代表有无模型，models_score是对应狼的模型们的分数
    len1 = len(xx)
    score_all=np.array(score_all)
    score_1=score_all
    num=0
    score=0
    for i in range(0, len1):
        #如果狼的这一属性离1（有）更近，则选择改基学习器
        x1 = abs(xx[i] - 1)
        x2 = abs(xx[i] - 0)
        if x1<x2:
            score=score+score_1[i]
            num=num+1
    if score==0:
        return 99999
    score_final=score/num

    return -score_final


# # 主程序
# func_details = ['F1', -100, 100, 30]
# function_name = func_details[0]
# Max_iter = 1000  # 迭代次数
# lb = 1  # 下界
# ub = 0  # 上届
# dim = 15  # 模型数量
# SearchAgents_no = 5  # 寻值的狼的数量
# models_score=[] #模型们的分数，是数组
# x = GWO(lb, ub, dim, SearchAgents_no, Max_iter,models_score)
