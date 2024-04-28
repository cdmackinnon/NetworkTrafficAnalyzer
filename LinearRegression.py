import pandas as pd
from sklearn.preprocessing import StandardScaler as SS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

def main():
    data = pd.read_csv('statistics_output.csv')
    X = data.drop(columns=['Website']).values
    y = data['Website'].values

    #Scale the data for regularization later
    ss = SS()
    X = ss.fit_transform(X)

    # Initialize the model with a maxmimum iteration size
    model = LogisticRegression(max_iter=1000)

    # Perform K-fold cross-validation
    train_scores = []
    test_scores = []
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model for each Kfold set
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))
    
    # Calculate and print average scores
    print("\nAverage Training Accuracy:", np.mean(train_scores))
    print("Average Testing Accuracy:", np.mean(test_scores))

main()