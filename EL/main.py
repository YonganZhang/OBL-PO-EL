# 导入普通模块
import warnings
import joblib

import numpy as np

from Z4predict import my_predict
from Z5other_models import other_models
from data_proccess import getpos

warnings.filterwarnings('ignore')
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
from matplotlib import rcParams

rcParams['axes.unicode_minus'] = False  # 正常显示符号
# 导入自己模块
from Z0datainput import datainput
from Z1distr import distr
from Z2modeltrain import modeltrain
from Z3evaluate import eva


## 做一些预备设置
K = 10  # 设置折数

## 0.导入数据+预处理
x1, y1, scale_x, scale_y = datainput()

# ## 1.总数据划分出：训练总集+测试集
# x_train_real, x_valid, y_train_real, y_valid, test_index, x_test, y_test, x_train, y_train = distr(x1, y1, K)
#
# 保存x
# joblib.dump(x_train_real, "./model/"+'x_train_real.pkl')
# joblib.dump(x_valid, "./model/"+'x_valid.pkl')
# joblib.dump(y_train_real, "./model/"+'y_train_real.pkl')
# joblib.dump(y_valid, "./model/"+'y_valid.pkl')
# joblib.dump(test_index, "./model/"+'test_index.pkl')
# joblib.dump(x_test, "./model/"+'x_test.pkl')
# joblib.dump(y_test, "./model/"+'y_test.pkl')
# joblib.dump(x_train, "./model/"+'x_train.pkl')
# joblib.dump(y_train, "./model/"+'y_train.pkl')
# 加载x
x_train_real =joblib.load("./model/"+'x_train_real.pkl')# 对于dataframe数据类型，也有自己的保存方法。
x_valid =joblib.load("./model/"+'x_valid.pkl')
y_train_real =joblib.load("./model/"+'y_train_real.pkl')
y_valid =joblib.load("./model/"+'y_valid.pkl')
test_index =joblib.load("./model/"+'test_index.pkl')
x_test =joblib.load("./model/"+'x_test.pkl')
y_test =joblib.load("./model/"+'y_test.pkl')
x_train =joblib.load("./model/"+'x_train.pkl')
y_train =joblib.load("./model/"+'y_train.pkl')


# 1.x_train_real,y_train_real是K折训练法里的训练集
# 2.x_valid，y_valid是K折训练法里的验证集
# 3.test_index是验证集在x_train里的序列（没确定），用来找K折的验证集长度用
# 4.x_test和y_test是测试集
# 5.x_train和y_train是除了测试集以外的训练集
## 2.模型训练
#zresult, result, models_str = modeltrain(K, x_train_real, x_valid, y_train_real, y_valid, scale_x, scale_y, )
# #
# # ## 3.计算误差值
# zzaeva=eva(K,test_index,result,models_str)

# 4.OBL-PO-EL
n=3
position = getpos(n)
position= np.array(position)
my_predict(n,K, position[0],y_test, scale_y, x_train, y_train, x_test,scale_x)
#
# ## 5.对其他模型进行预测
# other_models(K, y_test, scale_y, scale_x,x_train, y_train, x_test)
