import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def find_minimum_conponents(img):
    pca = PCA()
    pca.fit(img)
    cum_ratio = np.cumsum(pca.explained_variance_ratio_)
    return np.searchsorted(cum_ratio, 0.95)

def problem_1():
    img = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)
    components = find_minimum_conponents(img)
    print(f'Number of PCA componets for black and white image: {components}')

def problem_2():
    img = cv2.imread('rose.jpg') / 255
    components_list = [find_minimum_conponents(img[:, :, c]) for c in range(3)]
    components = max(components_list)
    print(f'Number of PCA componets for black and white image: {components}')

if __name__ == '__main__':
    problem_1()
    problem_2()
