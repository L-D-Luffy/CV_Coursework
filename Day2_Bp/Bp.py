import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np
import sklearn.linear_model
import matplotlib

def initialize(input_num, layer1_num, layer2_num):
    model = dict()
    np.random.seed(12)
    w1 = np.random.randn(input_num, layer1_num)
    b1 = np.random.randn(1, layer1_num)
    w2 = np.random.randn(layer1_num, layer2_num)
    b2 = np.random.randn(1, layer2_num)
    model['w1'] = w1
    model['b1'] = b1
    model['w2'] = w2
    model['b2'] = b2
    return model

def predict(model, x):
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def forward(model, x, y):
    epsilon = 0.043
    reg_lamda =0.0001
    x = x.reshape(1,2) #为什么要reshape?
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)#维度问题
    loss = -np.log(probs[0,y])

    #导数信息
    delta3 = probs
    delta3[0, y] -= 1#原来的代码时使用迭代器一起取数
    dw2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(w2.T)*(1-np.power(a1,2))
    dw1 = np.dot(x.T, delta2)

    db1 = np.sum(delta2, axis=0)

    dw2 += reg_lamda*w2
    dw1 += reg_lamda*w1

    w1 += -epsilon*dw1
    b1 += -epsilon * db1
    w2 += -epsilon * dw2
    b2 += -epsilon * db2

    model['w1'] = w1
    model['b1'] = b1
    model['w2'] = w2
    model['b2'] = b2
    return loss, model

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

def main():
    epochs = 200
    matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
    np.random.seed(3)
    X, Y = sklearn.datasets.make_moons(200, noise=0.20)
    print(len(X))
    model = initialize(2, 5, 2)
    for epoch in range(epochs):
        loss_ep = 0
        for i in range(len(X)):
            loss, model = forward(model, X[i], Y[i])
            loss_ep += loss
        loss_ep = loss_ep/len(X)
        if epoch%100==0:
            print('epoch:', epoch, 'loss:', loss_ep, '\n')
    plot_decision_boundary(lambda x: predict(model, x), X, Y)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=Y, cmap=plt.cm.Spectral)
    plt.show()

if __name__ == '__main__':
    main()