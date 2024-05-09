import pandas as pd
import matplotlib.pyplot as plt


#Read Data)
data = pd.read_csv('/Users/vedantbedi/Desktop/Desktop/Skills/Machine Learning/Linear Regression/test.csv')
plt.scatter(data.x, data.y)

#Checking Correlation
print(data.corr()) #Correlation is near 1, so we can use linear regression

#Cost-Functin
def cost_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x= points.iloc[i].x 
        y= points.iloc[i].y
        total_error += (y - (m*x + b))**2
    total_error/= float(len(points))       
    
def step_gradient(m_current, b_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_m, new_b]

m=0
b=0
learning_rate = 0.0001
for i in range(1000):
    m,b = step_gradient(m,b,data,learning_rate)

print(m,b)
plt.scatter(data.x, data.y, color='black')
plt.plot(data.x, m*data.x + b, color='red')
plt.show()
