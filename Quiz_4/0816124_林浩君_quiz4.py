import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error


def polyshape(d, x):
    return PolynomialFeatures(d).fit_transform(x.reshape(-1,1))

def make_predict_with_model(x, y, x_pred):
    lreg = LinearRegression(fit_intercept=False)
    lreg.fit(x, y)
    y_pred = lreg.predict(x_pred)
    return y_pred

def gen(degree, num_boot, size, x, y):
    predicted_values = []
    for i in range(num_boot):
        indexes=np.sort(np.random.choice(x.shape[0], size=size, replace=False))
        y_pred = make_predict_with_model(polyshape(degree, x[indexes]), y[indexes], polyshape(degree, x))
        predicted_values.append(y_pred)
    return predicted_values

def problem1():
    df = pd.read_csv("noisypopulation.csv")
    x, y, f = df.x, df.y, df.f
    
    _, axes=plt.subplots(figsize=(20,8), nrows=3, ncols=3)
    for i in range(3):
        axes[i][0].set_ylabel("$p_R$", fontsize=18)
        for j in range(3):
            axes[i][j].set_xlabel("$x$", fontsize=18)
            axes[i][j].set_ylim([0,1])
            axes[i][j].set_xlim([0,1])
    plt.tight_layout()

    degrees = [1, 3, 100]
    sample_size = [2, 39, 100]
    for i, degree in enumerate(degrees):
        for j, size in enumerate(sample_size):
            indexes=np.sort(np.random.choice(x.shape[0], size=size, replace=False))
            prediction = gen(degree, 200, size, np.array(x), np.array(y))

            axes[i][j].plot(x, f, label="f", color='darkblue', linewidth=4)
            axes[i][j].plot(x, y, '.', label="Population y", color='#009193', markersize=8)
            axes[i][j].plot(x[indexes], y[indexes], 's', color='black', label="Data y")
            for p in prediction[:-1]:
                axes[i][j].plot(x, p, alpha=0.03, color='#FF9300')
            axes[i][j].plot(x, prediction[-1], alpha=0.3, color='#FF9300', label=f"Degree: {degree} Sample size: {size}")
            axes[i][j].legend(loc='best')

    plt.show()

    print('Q1. Why the plot shows for example for sizesample one all horizon line because take one point as regression')
    print('Ans. Since we only choose one single point P(a, b), we can get zero mse by choosing regression function as y = b, which will always be horizon line.')

def reg_with_validation(rs, x, y, degree, alphas):
    x_train, x_val, y_train, y_val = train_test_split(np.array(x), np.array(y), train_size = 0.8, random_state=rs)
    training_error, validation_error = [],[]
    x_poly_train = PolynomialFeatures(degree).fit_transform(x_train.reshape(-1,1))
    x_poly_val= PolynomialFeatures(degree).fit_transform(x_val.reshape(-1,1))

    for alpha in alphas:
        ridge_reg = Ridge(alpha=alpha, fit_intercept=False)
        ridge_reg.fit(x_poly_train, y_train)
        y_train_pred = ridge_reg.predict(x_poly_train)
        y_val_pred = ridge_reg.predict(x_poly_val)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_val = mean_squared_error(y_val, y_val_pred) 
        training_error.append(mse_train)
        validation_error.append(mse_val)

    return training_error, validation_error

def reg_with_cross_validation(rs, df, degree, alphas):
    df_new = df.sample(len(df), random_state=rs)
    x = df_new[['x']].values
    y = df_new['y'].values

    training_error, validation_error = [],[]
    x_poly = PolynomialFeatures(degree).fit_transform(x.reshape(-1,1))
    for alpha in alphas:
        ridge_reg = Ridge(alpha=alpha, fit_intercept=False)
        ridge_cv = cross_validate(ridge_reg, x_poly, y, cv=5, return_train_score=True)
        mse_train = np.mean(ridge_cv['train_score'])
        mse_val = np.mean(ridge_cv['test_score'])    
        training_error.append(mse_train)
        validation_error.append(mse_val)

    return training_error, validation_error


def problem2():
    df = pd.read_csv("polynomial50.csv")
    x, y, f = df.x, df.y, df.f
    
    plt.figure()
    plt.title('Predictor vs Response plot')
    plt.scatter(x, y, color='blue', label='Observed values')
    plt.plot(x, f, color='black', label='True function')
    plt.xlabel('Predictor - X')
    plt.ylabel('Response - Y')
    plt.legend()

    ran_state = [0, 10, 21, 42, 66, 109, 310, 1969]
    alphas = [1e-7, 1e-5, 1e-3, 0.01, 0.1, 1]
    degree = 30

    best_alpha = []
    for i in range(len(ran_state)):
        training_error, validation_error = reg_with_validation(ran_state[i], x, y, degree, alphas)
        best_mse = np.min(validation_error)
        best_parameter = alphas[np.argmin(validation_error)]
        best_alpha.append(best_parameter)
        
        _, ax = plt.subplots(figsize = (6,4))
        ax.plot(alphas, training_error,'s--', label = 'Training error', color = 'Darkblue', linewidth=2)
        ax.plot(alphas, validation_error,'s-', label = 'Validation error', color ='#9FC131FF', linewidth=2)
        ax.axvline(best_parameter, 0, 0.75, color = 'r', label = f'Min validation error at alpha = {best_parameter}')
        ax.set_xlabel('Value of Alpha',fontsize=15)
        ax.set_ylabel('Mean Squared Error',fontsize=15)
        ax.set_ylim([0,0.010])
        ax.legend(loc = 'best',fontsize=12)
        bm = round(best_mse, 5)
        ax.set_title(f'Best alpha is {best_parameter} with MSE {bm}',fontsize=16)
        ax.set_xscale('log')
        plt.tight_layout()

    best_cv_alpha = []
    for i in range(len(ran_state)):
        training_error, validation_error = reg_with_cross_validation(ran_state[i], df, degree, alphas)
        best_mse  = np.min(validation_error)
        best_parameter = alphas[np.argmin(validation_error)]
        best_cv_alpha.append(best_parameter)

        _, ax = plt.subplots(figsize = (6,4))
        ax.plot(alphas, training_error,'s--', label = 'Training error', color = 'Darkblue', linewidth=2)
        ax.plot(alphas, validation_error,'s-', label = 'Validation error', color ='#9FC131FF', linewidth=2 )
        ax.axvline(best_parameter, 0, 0.75, color = 'r', label = f'Min validation error at alpha = {best_parameter}')
        ax.set_xlabel('Value of Alpha',fontsize=15)
        ax.set_ylabel('Mean Squared Error',fontsize=15)
        ax.legend(loc = 'best',fontsize=12)
        bm = round(best_mse, 5)
        ax.set_title(f'Best alpha is {best_parameter} with MSE {bm}',fontsize=16)
        ax.set_xscale('log')
        plt.tight_layout()
        
    plt.show()

problem1()
problem2()
