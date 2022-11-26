import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
import seaborn as sns

sns.set()
np.random.seed(816124)

def f1(x):
  a = 3.2
  b = 5.6
  beta = gamma(a) * gamma(b) / gamma(a + b)
  p = x ** (a - 1) * (1 - x) ** (b - 1)
  return 1/beta * p

def beta_gen(n):
  i = 0
  output = np.zeros(n)

  mode = (3.2-1)/(3.2+5.6-2) ## Mode of Beta distribution
  c = f1(mode)
  while i < n:
    U = np.random.uniform(size = 1)
    V = np.random.uniform(size = 1)
    if U < 1/c * f1(V):
      output[i] = V
      i = i + 1
  return output

# Generate a random varibale Y ~ beta(3.2, 5.6) by Accept/Reject algorithm
def problem_1():
    px = np.arange(0, 1 + 0.01, 0.01)
    py = f1(px)

    Y = beta_gen(n = 1000)
    _, ax = plt.subplots()
    ax.hist(Y,density=True)
    ax.plot(px,py)
    plt.title("Beta(3.2, 5.6) || Example 1")
    plt.show()

##Beta(2,6) Generation
def beta_gen2(n): 
  i = 0
  output = np.zeros(n)
  while i < n:
    U = np.random.uniform(size = 2 + 6)
    p1 = np.sum(np.log(U[0:2]))
    p2 = np.sum(np.log(U))
    output[i] = p1/p2
    i = i + 1
  return output

##PDF of Beta(2,6)
def f2(x):
  a = 2
  b = 6
  beta = gamma(a) * gamma(b) / gamma(a + b)
  p = x ** (a - 1) * (1 - x) ** (b - 1)
  return 1/beta * p

def beta_gen3(n):
  i = 0
  M = 1.67
  output = np.zeros(n)
  while i < n:
    U = np.random.uniform(size = 1)
    V = beta_gen2(1)
    if U < (1/M) * (f1(V) / f2(V)):
      output[i] = V
      i = i + 1
  return output
  
def problem_2():
    px = np.arange(0, 1 + 0.01, 0.01)
    py = f1(px)

    Y = beta_gen3(n = 1000)
    _, ax = plt.subplots()
    ax.hist(Y,density=True)
    ax.plot(px, py)
    plt.title("Beta(3.2, 5.6) || Example 2")
    plt.show()

def prob(A):
    """Computes the probability of a proposition, A."""    
    return round(A[A == True].count() / A.count() * 100, 3)

def conditional(proposition, given):
    """Probability of A conditioned on given."""
    return prob(proposition[given])

def problem_3():
    gss = pd.read_csv('gss.csv', index_col=0)

    banker = (gss['indus10'] == 6870)
    female = (gss['sex'] == 2)
    liberal = (gss['polviews'] <= 3)
    democrat = (gss['partyid'] <= 1)
    
    print(f"In this dataset, there are {banker.count()} bankers.")
    print(f"About {prob(banker)}% of the respondents work in banking, "
        + f"so if we choose a random person from the dataset, "
        + f"the probability they are a banker is about {prob(banker)}%.")
    print(f"If we choose a random person in this dataset, "
        + f"the probability they are liberal is about {prob(liberal)}%.")
    print(f"About {conditional(female, given=banker)}% of the bankers in this dataset are female.")
    print(f"About {conditional(liberal, given=female)}% of female respondents are liberal.")
    print(f"Only about {conditional(banker, given=female)}% of female respondents are bankers.")
    print(f"About {conditional(female, given=liberal & democrat)}% of liberal Democrats are female.")
    print(f"About {conditional(liberal & female, given=banker)}% of bankers are liberal women.")
    

if __name__ == '__main__':
    problem_1()
    problem_2()
    problem_3()
