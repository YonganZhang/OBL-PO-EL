from models13 import models15


def modeltrain(K,x_train_real, x_valid, y_train_real, y_valid, scale_x, scale_y):
    global models_str
    result = []
    zresult = []
    ttemp = []
    for i in range(K):
        print(i)
        # 将预测结果放到result中，列名分别为：模型名字、真实值、预测值，每次循环是一折中的结果
        reapend,models_str=models15(i,x_train_real[i], x_valid[i], y_train_real[i], y_valid[i], scale_x, scale_y)
        result.append(reapend)  # 1是训练分数，2是测试分数
    # 将训练结果放到zresult中，列名分别为：模型名字、真实值、预测值
    zresult.extend(result[0][0])
    zresult.extend(result[0][1])
    temp = [0] * len(result[3][2][0])
    for i in range(len(result[3][2][0])):
        for j in range(K):
            temp[i] = temp[i] + result[j][2][0][i]
    for i in range(len(result[0][2][0])):
        ttemp.append(temp[i] / K)
    zresult.extend([ttemp])
    return zresult,result,models_str