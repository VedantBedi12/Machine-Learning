import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('/Users/vedantbedi/Desktop/Desktop/Skills/Machine Learning/Linear Regression/Multiple LR/teams.csv')
#print(dataset.head())

def cost(x, y, w, b):
    m=len(x)
    cost=0
    for i in range(m):
        l=np.dot(w,x[i])+b
        cost+=(l-y[i])**2
    return cost/(2*m)

def gradient_descent(x, y, w, b, learning_rate):
    m = len(x)
    for i in range(m):
        l = np.dot(w, x[i]) + b - y[i]
        for j in range(len(w)):
            w[j] = w[j] - learning_rate * l * x[i][j]
        b = b - learning_rate * l
    return w, b

def predict(x, w, b):
    l=[]
    for i in range(len(x)):
        l.append(np.dot(w,x[i])+b)
    return l

dataset=dataset.drop(['team', 'year', 'age', 'height', 'weight'], axis =1)
#print(dataset.corr()["medals"])
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values
W_ini=np.array([0,0,0], dtype=object)
b_ini=1
m=len(X_train)
learning_rate=0.0000001
n_iterations=1000

cost2=cost(X_train, y_train, W_ini, b_ini)
print(cost2)

W_ini, b_ini = gradient_descent(X_train, y_train, W_ini, b_ini, learning_rate)
#print(W_ini, b_ini)

cost1=cost(X_train, y_train, W_ini, b_ini)
print(cost1)
y_pred=predict(X_train, W_ini, b_ini)
#print(y_pred)

plt.plot(range(m), y_train, color = 'red', label = 'Real data')
plt.plot(range(m), y_pred, color = 'blue', label = 'Predicted data')
plt.show()