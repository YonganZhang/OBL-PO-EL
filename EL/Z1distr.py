from sklearn.model_selection import train_test_split, KFold


def distr(x1,y1,K):
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=1)
    # K折训练法
    x_train_real = []
    x_valid = []
    y_train_real = []
    y_valid = []
    test_index = []
    kf = KFold(n_splits=K, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(x_train):  # 注意：需要保证数据量是K的倍数，不然下面会处理错误，因为每折的数据长度不一样
        x_train_real.append(x_train[train_index])
        y_train_real.append(y_train[train_index])
        x_valid.append(x_train[test_index])
        y_valid.append(y_train[test_index])
    return x_train_real, x_valid,y_train_real, y_valid,test_index,x_test, y_test,x_train, y_train