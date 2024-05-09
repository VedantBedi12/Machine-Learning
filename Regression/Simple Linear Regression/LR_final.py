import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cost(x,y,w,b):
    cos=0
    for i in range(len(x)):
        cos+=(w*x[i]+b-y[i])**2
    return cos/(2*len(x))

def gradient_descent(x,y,w,b,learning_rate,iterations):
    for i in range(iterations):
        c=w*x+b-y
        w=w-learning_rate*sum((c)*x)/len(x)
        b=b-learning_rate*sum(c)/len(x)
    return w,b

df=pd.read_csv('/Users/vedantbedi/Desktop/Desktop/Skills/Machine Learning/Linear Regression/Simple LR/test.csv')
#print(df.head())

x_train=df['x']
y_train=df['y']
#print(df.corr()['y'])
w_ini=0
b_ini=0
learning_rate=0.00001
iterations=1000
ini_cost=cost(x_train,y_train,w_ini,b_ini)
print(ini_cost)
w_ini,b_ini=gradient_descent(x_train,y_train,w_ini,b_ini,learning_rate,iterations)
print(w_ini,b_ini)
fin_cost=cost(x_train,y_train,w_ini,b_ini)
print(fin_cost)

y_pred=w_ini*x_train+b_ini
plt.scatter(x_train,y_train)
plt.plot(x_train,y_pred, c='r')
plt.show()