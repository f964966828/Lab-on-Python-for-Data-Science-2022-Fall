import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def problem1():
    df = pd.read_csv('CarPrice_Assignment.csv')

    print(df.columns)
    print(df.head(10))

    plt.figure()
    plt.scatter(df['horsepower'], df['price'])
    plt.show()
    
def problem2():
    df = pd.read_csv('CarPrice_Assignment.csv')
    x = df['horsepower']
    y = df['price']

    beta0 = -3000
    beta1_list = [i for i in range(251)]

    mse_list = list()
    for beta1 in beta1_list:
        pred_y = beta0 + beta1 * x
        mse = np.mean((y - pred_y) ** 2)
        mse_list.append(mse)

    plt.figure()
    plt.plot(beta1_list, mse_list)
    plt.show()
    

def problem3():
    df = pd.read_csv('CarPrice_Assignment.csv')
    x = df[['horsepower']]
    y = df['price']

    x_train, x_test, y_train, y_test = train_test_split(x , y, train_size=0.7)

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(mean_squared_error(y_test, y_pred))

    plt.figure()
    plt.scatter(x_test, y_test, label='data points')
    plt.plot(x_test, y_pred, 'r', label='model predictions')
    plt.legend()
    plt.show()
    
    
def problem4():
    df = pd.read_csv('CarPrice_Assignment.csv')
    x = df[['horsepower', 'peakrpm', 'citympg', 'highwaympg']]
    y = df['price']

    x_train, x_test, y_train, y_test = train_test_split(x , y, train_size=0.7)

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(mean_squared_error(y_test, y_pred))
    
    
problem1()
problem2()
problem3()
problem4()