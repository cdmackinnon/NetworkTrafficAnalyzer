import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

def main():
    # Read in data after preprocessing
    data = pd.read_csv('statistics_output.csv')
    X = data.drop(columns=['Website'])
    y = data['Website']

    # Train Test Split 10% of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    print("TTS Accuracy:", model.score(X_test, y_test))
    print("\n")

    #Visualize Tree
    plt.figure(figsize=(20,10))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_)
    plt.show()

    #K-fold Validation
    print("50 Fold Cross-validation average testing score: ",np.mean(cross_validation(X,y)))
    print("\n")


def cross_validation(X, y):
    test_accuracies = []
    test_scores = []
    model = DecisionTreeClassifier()
    kf = KFold(n_splits=50, shuffle=True)
    X = X.values
    y = y.values
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        test_scores.append(model.score(X_test, y_test))
        test_accuracies.append(np.mean(test_scores))
    return test_accuracies
