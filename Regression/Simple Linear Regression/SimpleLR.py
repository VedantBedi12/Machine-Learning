import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define functions for intercept and slope calculation
def intercept(X, Y):
    return np.mean(Y) - slope(X, Y) * np.mean(X)

def slope(X, Y):
    return np.sum((X - np.mean(X)) * (Y - np.mean(Y))) / np.sum((X - np.mean(X)) ** 2)

# Read Data
data = pd.read_csv('/Users/vedantbedi/Desktop/Desktop/Skills/Machine Learning/Linear Regression/test.csv')
x = np.array(data['x'])
y = np.array(data['y'])

plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')


#To check if there are many missing values
#print("Missing values in X:", np.isnan(x).sum())
#print("Missing values in Y:", np.isnan(y).sum())


# Calculate slope and intercept
B1 = slope(x, y)
B0 = intercept(x, y)
print("Intercept: ", B0)
print("Slope: ", B1)

y_pred = B0 + B1 * x # Final predicted values for linear function
#print("Predicted Y: ", y_pred)
plt.plot(x, y_pred, color='red')
plt.show()

# Calculate R^2
y_mean = np.mean(y)
SSR = np.sum((y_pred - y_mean) ** 2)
SST = np.sum((y - y_mean) ** 2)
R2 = SSR / SST
print("R^2: ", R2)

# Calculate RMSE
RMSE = np.sqrt(np.sum((y - y_pred) ** 2) / len(y))
print("RMSE: ", RMSE)

# Calculate MAE
MAE = np.sum(np.abs(y - y_pred)) / len(y)
print("MAE: ", MAE)
