from sklearn import metrics

from save_excel import write_excel_xls


def evaluate(y_train, y_test):

    a1=metrics.mean_absolute_error(y_train, y_test)
    a2=metrics.mean_squared_error(y_train, y_test)
    #a3=metrics.mean_squared_log_error(y_train, y_test)
    a4=metrics.median_absolute_error(y_train, y_test)
    a5=metrics.r2_score(y_train, y_test)
    return [a1,a2,a4,a5]

def eva(K,test_index,result,models_str):
    # 将误差放到zzeva中，第一层为K折数、第二层为每折里的39个模型，第三层为每个模型的四个评估值
    zzeva = []
    for i in range(0, K):
        zeva = []
        aaa=len(result[1][1][0])
        for j in range(0, len(result[1][1][0]), len(test_index)):
            zeva.append(evaluate(result[i][1][0][j:j + len(test_index)], result[i][2][0][j:j + len(test_index)]))
        zzeva.append(zeva)
    # 计算K折的评估平均值
    temp=[]
    zaeva = [[], []]
    zaeva[0].extend([models_str])
    for j in range(0, len(zzeva[0])):
        temp = 0
        for i in range(0, len(zzeva)):  # 每一折的结果相加再除以折数
            temp = zzeva[i][j][3] + temp
        temp = temp / K
        zaeva[1].extend([temp])

    ##  开始写入R2（测试集）
    import xlwt
    f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表
    for i in range(len(models_str)):
        sheet1.write(i, 0, models_str[i])  # 写入数据参数对应 行, 列, 值
        sheet1.write(i, 1, zaeva[1][i])  # 写入数据参数对应 行, 列, 值
    f.save('模型们的平均预测精度（R2）.xls')  # 保存.xls到当前工作目录
    ##  开始写入泰勒图数据
    f1 = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
    sheet1 = f1.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表

    taile_result=[[],[],[]]
    for i in range(len(result)):
        taile_result[0].extend(result[i][0][0])
        taile_result[1].extend(result[i][1][0])
        taile_result[2].extend(result[i][2][0])
    write_excel_xls("泰勒图.xlsx","导出",taile_result)
    ## return score_all_3, score_all_2 #3为训练集的R2,2为测试集的全部指标

    return zaeva