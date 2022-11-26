import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def problem_1():
    
    # (a) Lex X ~ N(500, 75^2). Determine P(X>=600)
    prob = 1 - norm.cdf(600, 500, 75)
    print(f"(a). prob: {prob}")

    # (b) Plot the normal distribution of X ~ N(500, 75^2)
    mu = 500
    sigma = 75
    x = np.arange(200,800)
    prob = norm.pdf(x, mu, sigma)

    plt.figure()
    plt.plot(x, prob)
    plt.title(r'$\mathrm{N(\mu=500, \sigma^2=75^2)}$')
    plt.ylim((0,0.006))

    # (c) Calculating simple likelihoods
    x = [3, 5, 10]
    sigma = 2
    mu = np.arange(4, 8, 0.01)
    like = norm.pdf(x[0],mu,sigma)*norm.pdf(x[1],mu,sigma)*norm.pdf(x[2],mu,sigma)

    plt.figure()
    plt.plot(mu,like,color="darkred")
    plt.title('Likelihood Function')
    plt.xlabel(r'$\mu$')

    # (d) Determine the maximum likelihood estimate for mu
    mle = mu[np.argmax(like)]
    print(f"(d). mle: {mle}")

    # (e) Find the CDF from the PDF figure
    cdf_prob = 1 - prob[:600 - 200].sum()
    print(f"(e). prob: {cdf_prob}")

    # (f) plot your CDF
    x = np.arange(200,800)
    cdf = [prob[:i - 200].sum() for i in range(200, 800)]

    plt.figure()
    plt.plot(x, cdf)
    plt.title(r'$\mathrm{N(\mu=500, \sigma^2=75^2)}$')

    # Question
    print("Question: How would you numerically maximize this function if both the mean and variance were unknown? How would you visualize the likelihood function?")
    print("Answer: Search both mu and sigma by declare two possible value arrays, and visualize this 2D likelihood function as heat map where colors on the graph indicates likelihood value.")

    plt.show()

def bootstrap(df):
    selectionIndex = np.random.randint(len(df), size = len(df))
    new_df = df.iloc[selectionIndex]
    return new_df

def problem_2():
    df = pd.read_csv('Advertising_adj.csv')
    beta0_list, beta1_list = [],[]
    number_of_bootstraps = 1000

    for i in range(number_of_bootstraps):
        df_new = bootstrap(df)
        x = df_new.tv
        y = df_new.sales

        xmean = x.mean()
        ymean = y.mean()
        
        beta1 = ( (x-xmean)*(y-ymean) ).sum() / ( (x-xmean)*(x-xmean) ).sum()
        beta0 = ymean - beta1 * xmean

        beta0_list.append(beta0)
        beta1_list.append(beta1)


    fig, ax = plt.subplots(1, 2, figsize=(18,8))
    ax[0].hist(beta0_list)
    ax[1].hist(beta1_list)
    ax[0].set_xlabel('Beta 0')
    ax[1].set_xlabel('Beta 1')
    ax[0].set_ylabel('Frequency')

    plt.show()

def plot_simulation(simulation, confidence):
    plt.figure()
    plt.hist(simulation, bins = 30, label = 'beta distribution', align = 'left', density = True)
    plt.axvline(confidence[1], 0, 1, color = 'r', label = 'Right Interval')
    plt.axvline(confidence[0], 0, 1, color = 'red', label = 'Left Interval')
    plt.xlabel('Beta value')
    plt.ylabel('Frequency')
    plt.title('Confidence Interval')
    plt.legend(frameon = False, loc = 'upper right')

def problem_3():
    df = pd.read_csv('Advertising_adj.csv')
    beta0_list, beta1_list = [],[]
    numberOfBootstraps = 100

    for i in range(numberOfBootstraps):
        df_new = bootstrap(df)
        xmean = df_new.tv.mean()
        ymean = df_new.sales.mean()
        
        beta1 = np.dot((df_new.tv-xmean) , (df_new.sales-ymean))/((df_new.tv-xmean)**2).sum()
        beta0 = ymean - beta1*xmean
        
        beta0_list.append(beta0)
        beta1_list.append(beta1)
    
    beta0_list.sort()
    beta1_list.sort()

    beta0_CI = (np.percentile(beta0_list, 2.5) , np.percentile(beta0_list, 97.5))
    beta1_CI = (np.percentile(beta1_list, 2.5) , np.percentile(beta1_list, 97.5))

    print(f'The beta0 confidence interval is {beta0_CI}')
    print(f'The beta1 confidence interval is {beta1_CI}')

    plot_simulation(beta0_list, beta0_CI)
    plot_simulation(beta1_list, beta1_CI)

    plt.show()

if __name__ == "__main__":
    problem_1()
    problem_2()
    problem_3()
