import numpy as np
import matplotlib.pyplot as plt
#对每一个样本，更新一次W，并同时返回误差
def linear_pre(W,X1,Y):
    learn_rate = 0.16
    Result = X1.dot(W.T)
    if Result >0:
        Ypre = 1
    else:
        Ypre = 0
    Wnew = W + learn_rate*(Y-Ypre)*X1
    error = Y - Ypre
    return Wnew,error

np.random.seed(12)

num_observations = 500

x1 = np.random.multivariate_normal([0,0],[[1,.75],[.75,1]],num_observations)
x2 = np.random.multivariate_normal([1,4],[[1,.75],[.75,1]],num_observations)

X = np.vstack((x1,x2)).astype(np.float32)
Y = np.hstack((np.zeros(num_observations),np.ones(num_observations)))

X1 = np.hstack((X,np.ones([2*num_observations,1])))
W = np.random.randn(1,3)

error_sum = 0
gamma = 0.00001
#这里想加上那个误差范围的判断，但一开始误差都是0，所以就设定让它在500次之后再生效了，
#你们有好的写法就改一下
for i in range(len(X1)):
    W,error = linear_pre(W, X1[i], Y[i])
    error_sum += error
    if (error_sum/(i+1) < gamma)&(i>500):
        break


f1=plt.figure(1)

plt.scatter(X[0:500,0],X[0:500,1],c = 'red')
plt.scatter(X[500:,0],X[500:,1],c ='green')

x_line = np.linspace(-3,3,1000)
y_line = -((W[0,0]*x_line+W[0,2])/W[0,1])

plt.plot(x_line,y_line,color = 'black')

plt.show()







