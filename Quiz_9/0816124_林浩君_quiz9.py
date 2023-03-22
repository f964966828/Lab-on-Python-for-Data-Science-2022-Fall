from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    mnist = fetch_openml(data_id=554)

    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1/7)
    print(f"Training dataset: {X_train.shape}, Test dataset: {X_test.shape}")

    clf = LogisticRegression(max_iter=1000, n_jobs=5)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_pred)
    print(f'Accuracy: {acc}')

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    plt.figure(figsize=(12,12))
    sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', fmt='0.4g')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(acc)
    plt.title(all_sample_title)
    plt.show()
