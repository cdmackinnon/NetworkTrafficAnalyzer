import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['Website']).values
    y = data['Website'].values
    return X, y

# Scale the data using StandardScaler.
def scale_data(X):
    scaler = SS()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Perform K-fold cross-validation for different values of C
def cross_validation(X, y, C_values):
    test_accuracies = []
    for C in C_values:
        model = SVC(kernel='linear', C=C)
        test_scores = []
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            test_scores.append(model.score(X_test, y_test))
        test_accuracies.append(np.mean(test_scores))
    return test_accuracies

#Visualize different accuracies of C
def plot_results(C_values, test_accuracies):
    plt.figure(figsize=(8, 8))
    plt.plot(C_values, test_accuracies, color='r', label='Testing')
    plt.xlabel('$C$', fontsize=14)
    plt.ylabel('Average Testing Accuracy', fontsize=14)
    plt.title('Average Testing Accuracy vs. $C$', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()

# Select Highest C Value
def find_best_C(C_values, test_accuracies):
    best_C_index = np.argmax(test_accuracies)
    best_C = C_values[best_C_index]
    best_accuracy = test_accuracies[best_C_index]
    return best_C, best_accuracy

def visualize_pca(X, y, best_C):
    X_pca = PCA(n_components=2).fit_transform(X)
    y_encoded = LabelEncoder().fit_transform(y)
    model = SVC(kernel='linear', C=best_C).fit(X_pca, y_encoded)

    h = .02; x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.4)  # Adjust the alpha value here
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap=plt.cm.Paired, alpha=0.9) # Adjust the alpha value here
    plt.xlabel('Principal Component 1'); plt.ylabel('Principal Component 2')
    plt.title('SVM Decision Boundaries (Graphed After PCA)')

    original_target_names = np.unique(y)
    handles, labels = scatter.legend_elements()
    plt.legend(handles, original_target_names, loc="lower left", title="Classes")
    plt.show()


def main():
    file_path = 'statistics_output.csv'
    X, y = load_data(file_path)

    # Scale data
    X_scaled, scaler = scale_data(X)

    # Choose C values for cross-validation
    C_values = np.linspace(4.5, 5.5, 10)
    test_accuracies = cross_validation(X_scaled, y, C_values)

    # Plot results and store best C value and accuracy
    plot_results(C_values, test_accuracies)
    best_C, best_accuracy = find_best_C(C_values, test_accuracies)
    print("Best C value:", best_C)
    print("Corresponding Testing Accuracy:", best_accuracy)

    # Visualize PCA after selecting the best C value
    visualize_pca(X_scaled, y, best_C)

    return X, y, best_C

main()
