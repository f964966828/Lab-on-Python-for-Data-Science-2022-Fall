import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from typing import Union

def find_minimum_conponents(A:np.ndarray, rs:int):
    pca = PCA(random_state=rs)
    pca.fit(A)
    cum_ratio = np.cumsum(pca.explained_variance_ratio_)
    return min(np.searchsorted(cum_ratio, 0.95) + 1, A.shape[1])

def TASK1(A:np.ndarray, rs:int):
    
    #Complete this function----------------------------
    
    #   rs is random_state, use it for PCA
    
    #   X = matrix of SVDs that descibes >95% of image (as close to 95% as possible)
    #   X.ndim = 2
    
    #--------------------------------------------------
    
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(A)

    mc = find_minimum_conponents(A_scaled, rs)
    pca = PCA(n_components=mc, random_state=rs)
    pca.fit(A_scaled)
    X = pca.transform(A_scaled)

    return pca, X

def TASK2(pca, X:np.ndarray):
    
    #Complete this function----------------------------

    #   Note
    #   pca is an instance of the PCA class from the sklearn.decomposition module that you used in TASK1
    #   just pass it to this function
    
    #   X is also an output from function 1
    
    #   A_hat = compress matrix A from TASK1
    #   A_hat.ndim = 2
    #   A.shape == A_hat.shape
    
    #--------------------------------------------------
    
    A_hat = pca.inverse_transform(X)
    return A_hat


def TASK3(classifier, flag = False):

    #Complete this function----------------------------
    
    #   classifier input argument is a classification model from sklearn (can be KN, DT, LR)
    #   Slice iris dataset using columns 0 and 2, you should end up with another dataframe
    #   You need to train your classifier
    #   Create a mesh of points to plot the decision boundary
    #   Predict the class for each point in the mesh
    
    #--------------------------------------------------
    
    iris = datasets.load_iris()
    x, y = iris.data[:, :2], iris.target
    
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    classifier.fit(x, y)
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    #Z is a predicted variable
    Z = Z.reshape(xx.shape)
    
    # If you want to check your answer you can change flag to True
    # Plot the decision boundary and the data points
    if flag:
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.8)
        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.title(str(type(classifier)))
        plt.show()
    
    return xx, yy, Z


def TASK4(t_size:float, rs:int, n_estimators:int, lr:float, md:int):
  
    #Complete this function----------------------------
    
    #   t_size is test_size use for train_test_split
    
    #   n is n_estimators
    
    #   rs is random_state use this for train_test_split, and GBC
    
    #   md is max_depth
    
    #   lr is learning_rate
    
    #   accuracy should be between 0 and 1 (in form of decimal fractions)
    #   where 0 corresponds to 0% accuracy and 1 to 100% accuracy
    #   example: accuracy = 0.87 corresponds to 87%
    
    #--------------------------------------------------
    
    #   Load dataset
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=rs)
    
    
    # Declare classifier using function input arguments, don't forget to add other arguments
    gbc = GradientBoostingClassifier(
        random_state=rs, 
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=md   
    )

    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    
    return accuracy

