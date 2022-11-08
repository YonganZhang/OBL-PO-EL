from lazypredict.Supervised import LazyRegressor
import warnings


# filter warnings
warnings.filterwarnings('ignore')
# 正常显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示符号
from matplotlib import rcParams
rcParams['axes.unicode_minus']=False

def models41(x_train1, x_test1, y_train1, y_test1):
## 1.导入数据

## 开始41模型挑选
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(x_train1, x_test1, y_train1, y_test1)

    print(models)