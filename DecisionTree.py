import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Read in data after preprocessing
data = pd.read_csv('statistics_output.csv')
X = data.drop(columns=['Website'])
y = data['Website']

# Train Test Split 10% of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

#Visualize Tree
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_)
plt.show()