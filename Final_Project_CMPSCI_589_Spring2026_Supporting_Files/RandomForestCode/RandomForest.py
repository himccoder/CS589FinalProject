# RandomForest.py
import numpy as np
import math
from collections import Counter
from DecisionTree import DecisionTree


class RandomForest:
    """
    Random Forest classifier built on top of our Decision Tree implementation.

    Key ideas:
    - Each tree is trained on a bootstrap sample (sampling with replacement).
    - At each split node, only m = sqrt(n_features) randomly chosen attributes
      are considered. This is handled inside DecisionTree via the 'm' parameter.
    - Final predictions are made by majority vote across all trees.
    """

    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, min_gain=0.0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.trees = []
        self.numerical_attrs = set()

    def fit(self, X, y, numerical_attrs=None):
        """
        Train the random forest on (X, y).

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
        y : numpy array of shape (n_samples,)
        numerical_attrs : list or set of attribute indices that are numerical
        """
        if numerical_attrs is not None:
            self.numerical_attrs = set(numerical_attrs)
        else:
            self.numerical_attrs = set()

        n_samples, n_features = X.shape
        all_attributes = list(range(n_features))

        # m = sqrt(total features), rounded down. At least 1.
        m = max(1, int(math.sqrt(n_features)))

        self.trees = []
        for _ in range(self.n_trees):
            # Create a bootstrap sample by sampling n_samples rows with replacement
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[boot_indices]
            y_boot = y[boot_indices]

            # Build a decision tree. The 'm' parameter tells it to randomly
            # sample m attributes at each node before finding the best split.
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_gain=self.min_gain,
                m=m
            )
            tree.fit(X_boot, y_boot, all_attributes, numerical_attrs=self.numerical_attrs)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict class labels for each row in X using majority voting.
        """
        # Collect predictions from every tree: shape (n_trees, n_samples)
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # For each sample, pick the most common label across all trees
        final_preds = []
        for i in range(X.shape[0]):
            votes = all_predictions[:, i]
            majority = Counter(votes).most_common(1)[0][0]
            final_preds.append(majority)

        return np.array(final_preds)
