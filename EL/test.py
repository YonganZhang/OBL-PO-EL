# import numpy as np
# from sklearn.preprocessing import StandardScaler as SS
#
#
# scale_x=SS()
# scale_y=SS()
#
# xw=[[1,2,0],[0,-1,1],[-1,0,2]]
# #xw=np.array(x_dum).reshape(-1,1)
#
# y=[0,5,10]
# y=np.array(y).reshape(-1,1)
#
#
#
# x1=scale_x.fit_transform(xw)
# #x1=x1.ravel()
#
# y1=scale_y.fit_transform(y)
# y1=y1.ravel()
#
# print(x1)
# print(y1)
# print("          ")
#
#
# print(scale_x.inverse_transform(x1)) # 逆标准化
# print(scale_y.inverse_transform(y1)) # 逆标准化


# from sklearn.model_selection import KFold
import numpy as np
birth_year = input()

# 输入一个一维数组
arr = input()
# 将输入每个数以空格键隔开做成数组
nums = [int(n) for n in arr.split()]

data= nums
posi1=[0]*len(data)
posi2=[0]*len(data)
leng=len(data)
for i in range(leng):
    j=i-1
    if (j<0):
        posi1[i]=0
        continue
    if (data[i]>=data[j]):
        posi1[i] = posi1[j] + 1
    else:
        posi1[i] = 0
for i in range(leng-1,0,-1):
    j = i + 1
    if (j > leng-1):
        posi2[i] = 0
        continue
    if (data[i] >= data[j]):
        posi2[i] = posi2[j] + 1
    else:
        posi2[i] = 0
        
a = np.array(posi1) + np.array(posi2)
ans=np.max(a)+1
print(ans)

# check scikit-learn version
# import sklearn
# print(sklearn.__version__)
# model=SVR()
# wrapper = MultiOutputRegressor(model)
#
# wrapper.fit(x, y)

#
#
# kf = KFold(n_splits = 5, shuffle=True, random_state=0)
# for train_index, test_index in kf.split(data):
#     print(train_index)
#     print(test_index)


# def solve(n):
#   sieve = [True] * (n + 1)
#   primes = []
#   for i in range(2, n + 1):
#      if sieve[i]:
#         primes.append(i)
#         for j in range(i, n + 1, i):
#            sieve[j] = False
#   return primes
#
#
# result=solve(5)
# z=""
# for i in range(len(result)):
#     z=z+str(result[i])
# print(int(z))


