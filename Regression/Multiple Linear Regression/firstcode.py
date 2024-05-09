import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/vedantbedi/Desktop/Desktop/Skills/Machine Learning/Linear Regression/Multiple LR/teams.csv')
#print(df.head())
df=df.drop(['team'], axis =1)
#we use correlation to find the relationship between the variables. If it is 1, it means that the variables are directly proportional to each other. If it is -1, it means that the variables are inversely proportional to each other. If it is 0, it means that the variables are independent of each other.
#we should only 1 dependent variable and multiple independent variables
#remove every value which has corr value almost equal to 0
df=df.drop(['weight', 'age', 'year', 'height'], axis =1)
print(df.corr()["medals"])


#show a graph bw variables to ensure that they are linearly relatedx`x`
#plt.scatter(df['medals'], df['events'])
#plt.show()
#sns.pairplot(df) #idealy graph should be linea exxcept for the dependent variable
#plt.show()

df_np = df.to_numpy()
#print(df_np.shape) finds shape of array
x_train, y_train = df_np[:, :3], df_np[:, -1]
#print(x_train.shape, y_train.shape)
#print(x_train)
#print(y_train)


#check null columns- there should ideally be no null columns
#print(df.isnull().sum())

class MLR:
    def __init__(self, learning_rate = 0.01, n_interations=1000):
        self.learning_rate = learning_rate
        self.n_interations = n_interations
        self.weights = None
        self.bias = None
        self.loss=[]
    
    
    def _mean_squared_error(self, y, y_hat):
        error = 0
        for i in range(len(y)):
            error += (y[i] - y_hat[i])**2
        return error/len(y)
    
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(self.n_interations):
            #y = mx + c
            y_hat = np.dot(X, self.weights) + self.bias
            loss = self._mean_squared_error(y, y_hat)
            self.loss.append(loss)
            
            #calculate partial derivatives
            dw = (1/n_samples) * np.dot(X.T, (y_hat-y))
            db = (1/n_samples) * np.sum(y_hat-y)
            
            #update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        y_hat = np.dot(X, self.weights) + self.bias
        return y_hat
        


model = MLR(0.5,100)
model.fit(x_train, y_train)
y_pred = model.predict(x_train)
print(y_pred)
#print(y_train)
#print(model.loss)
#plt.plot(range(model.n_interations), model.loss)
#plt.xlabel('Iterations')
#plt.ylabel('Loss')
#plt.show()



