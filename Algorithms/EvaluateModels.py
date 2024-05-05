import pandas as pd
from sklearn.preprocessing import StandardScaler as SS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from IPython.display import display

def main():
    # Read in data after preprocessing
    data = pd.read_csv('statistics_output.csv')
    X = data.drop(columns=['Website'])
    y = data['Website']

    # Generate scaled data for svm and logistic regression
    ss = SS()
    Xscaled = ss.fit_transform(X)

    # Logistic regression
    logistic_regression_5fold = cross_validation(Xscaled,y,5,LogisticRegression(max_iter=1000))
    logistic_regression_loo = cross_validation(Xscaled,y,50,LogisticRegression(max_iter=1000))

    # SVM C=5.27777 determined from SVM.py
    svm_5fold = cross_validation(Xscaled,y,5,SVC(kernel='linear',C=5.277777)) 
    svm_loo = cross_validation(Xscaled,y,50,SVC(kernel='linear',C=5.277777))

    # Decision Tree
    dtc_5fold = cross_validation(X.values,y.values,5,DecisionTreeClassifier())
    dtc_loo = cross_validation(X.values,y.values,50,DecisionTreeClassifier())

    df = pd.DataFrame(data=[logistic_regression_5fold,logistic_regression_loo,svm_5fold,svm_loo,dtc_5fold,dtc_loo],index=["LR 5-Fold","LR 50-Fold", "SVM 5-Fold", "SVM 50-Fold","DTC 5-Fold","DTC 50-Fold"], columns=["Average Internal Accuracy","Average External Accuracy"])
    fix,ax = plt.subplots()
    ax.axis('off')
    table = pd.plotting.table(ax,df,loc='center',cellLoc='center')
    plt.show()

def cross_validation(X, y, k, model):
    train_accuracies = []
    test_accuracies = []
    kf = KFold(n_splits=k, shuffle=True)
    X = X
    y = y
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        train_accuracies.append(model.score(X_train, y_train))
        test_accuracies.append(model.score(X_test, y_test))
    return (np.mean(train_accuracies),np.mean(test_accuracies))