import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\pandu\OneDrive\Documents\AllSem4\MLsem4\mllabsession03\DCT_withoutduplicate 6 1 1.csv"
df = pd.read_csv(file_path)

# Split features and target
X = df.drop(columns=["LABEL"]).values  # Extracting features
y = df["LABEL"].values  # Extracting target variable

# Function to compute entropy
def compute_entropy(labels):
    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

entropy_val = compute_entropy(y)
print(f"Entropy: {entropy_val}")

# Function to compute Gini index
def compute_gini(labels):
    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    return 1 - np.sum([p ** 2 for p in probabilities])

gini_val = compute_gini(y)
print(f"Gini Index: {gini_val}")

# Function to find the best feature to split on using Information Gain
def find_best_split(X, y):
    base_entropy = compute_entropy(y)
    info_gains = []
    
    for feature in range(X.shape[1]):
        unique_values = np.unique(X[:, feature])
        weighted_entropy = sum([(np.sum(X[:, feature] == v) / len(y)) * compute_entropy(y[X[:, feature] == v]) for v in unique_values])
        info_gains.append(base_entropy - weighted_entropy)
    
    return np.argmax(info_gains)

best_feature_index = find_best_split(X, y)
print(f"Best feature to split: {best_feature_index}")

# Function to bin continuous features
def discretize_feature(X, bins=4, strategy='uniform'):
    binning = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
    return binning.fit_transform(X.reshape(-1, 1)).astype(int)

# Example binning on the first feature
binned_first_feature = discretize_feature(X[:, 0])
print(f"Sample Binned Feature: {binned_first_feature[:10].flatten()}")

# Function to train a Decision Tree model
def train_decision_tree(X, y):
    tree_model = DecisionTreeClassifier(criterion='entropy')
    tree_model.fit(X, y)
    return tree_model

decision_tree = train_decision_tree(X, y)

# Function to visualize Decision Tree
def display_decision_tree(model, feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=feature_names)
    plt.show()

# Naming features
feature_labels = [f'Feature {i}' for i in range(X.shape[1])]
display_decision_tree(decision_tree, feature_labels)

# Function to plot decision boundary
def plot_tree_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.show()

# Execute decision boundary visualization if there are at least two features
if X.shape[1] >= 2:
    plot_tree_decision_boundary(X[:, :2], y, decision_tree)
