# DecisionTree.py
import numpy as np
from collections import Counter
import heapq


class DecisionTreeNode:
    def __init__(self, attribute=None, threshold=None, label=None):
        self.attribute = attribute   # index of the attribute to split on (None if leaf)
        self.threshold = threshold   # threshold value for numerical splits (None for categorical)
        self.label = label           # class label for leaf nodes
        self.children = {}           # categorical: {value: node}, numerical: {'<': node, '>=': node}


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=0.0,
                 purity_threshold=None, m=None):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.purity_threshold = purity_threshold
        self.m = m                   # number of random features to try at each split (for random forests)
        self.numerical_attrs = set() # set of attribute indices that are numerical

    def fit(self, X, y, attributes, numerical_attrs=None):
        if numerical_attrs is not None:
            self.numerical_attrs = set(numerical_attrs)
        else:
            self.numerical_attrs = set()
        self.root = self.build_tree(X, y, list(attributes), depth=0)

    def build_tree(self, X, y, attributes, depth=0):
        # Base case: all instances have the same label
        unique_labels = set(y)
        if len(unique_labels) == 1:
            return DecisionTreeNode(label=y[0])

        # Purity threshold stopping criterion
        if self.purity_threshold is not None:
            label_counts = Counter(y)
            majority_label, majority_count = label_counts.most_common(1)[0]



            if (majority_count / len(y)) >= self.purity_threshold:
                return DecisionTreeNode(label=majority_label)

        # No attributes left to split on
        if len(attributes) == 0:
            return DecisionTreeNode(label=Counter(y).most_common(1)[0][0])

        # Maximum depth stopping criterion
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode(label=Counter(y).most_common(1)[0][0])

        # Minimum samples stopping criterion
        if len(y) < self.min_samples_split:
            return DecisionTreeNode(label=Counter(y).most_common(1)[0][0])

        # Choose which attributes to consider at this node.
        # When m is set (Random Forest mode), sample m attributes at random.
        if self.m is not None and len(attributes) > self.m:


            candidate_attrs = list(np.random.choice(attributes, size=self.m, replace=False))
        else:
            candidate_attrs = list(attributes)

        # Find the best attribute (and threshold if numerical)
        best_attr, best_threshold, best_gain = self.best_attribute(X, y, candidate_attrs)

        # Minimum gain stopping criterion
        if best_attr is None or best_gain <= self.min_gain:
            return DecisionTreeNode(label=Counter(y).most_common(1)[0][0])

        node = DecisionTreeNode(attribute=best_attr, threshold=best_threshold)

        if best_attr in self.numerical_attrs:
            # Binary split for numerical attributes: left = < threshold, right = >= threshold
            x_col = X[:, best_attr].astype(float)
            left_mask = x_col < best_threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                return DecisionTreeNode(label=Counter(y).most_common(1)[0][0])

            remaining = [a for a in attributes if a != best_attr]


            node.children['<'] = self.build_tree(X[left_mask], y[left_mask], remaining, depth + 1)
            node.children['>='] = self.build_tree(X[right_mask], y[right_mask], remaining, depth + 1)

        else:
            # Multi-way split for categorical attributes
            attribute_values = set(X[:, best_attr])

            remaining = [a for a in attributes if a != best_attr]

            for val in attribute_values:
                mask = X[:, best_attr] == val
                child_X, child_y = X[mask], y[mask]

                if len(child_y) == 0:
                    node.children[val] = DecisionTreeNode(label=Counter(y).most_common(1)[0][0])
                else:
                    node.children[val] = self.build_tree(child_X, child_y, remaining, depth + 1)

        return node

    # Entropy and Information Gain (categorical attributes) -------------------------

    def entropy(self, y):
        label_counts = Counter(y)
        total = len(y)
        ent = 0.0
        for count in label_counts.values():
            p = count / total
            ent -= p * np.log2(p + 1e-9)
        return ent

    def information_gain(self, X_col, y):
        """Information gain for a categorical attribute."""
        original_entropy = self.entropy(y)
        total = len(y)
        weighted_entropy = 0.0
        for val in set(X_col):
            subset_y = y[X_col == val]
            w = len(subset_y) / total
            weighted_entropy += w * self.entropy(subset_y)
        return original_entropy - weighted_entropy

    # Vectorized threshold search for numerical attributes -------------------------------

    def find_best_threshold(self, x_col_float, y):
        """
        Find the split threshold that maximises information gain for a
        numerical attribute using a fully-vectorised cumulative-sum approach.

        This method works for binary or multi-class labels.  For binary labels
        (the common case) all computations are done in numpy without any Python
        loops over individual thresholds, making it orders-of-magnitude faster
        than the naive approach.
        """
        # Encode string labels to integers so we can use np.bincount
        unique_labels, y_int = np.unique(y, return_inverse=True)
        n_classes = len(unique_labels)

        n = len(y_int)
        sort_idx = np.argsort(x_col_float, kind='stable')
        x_sorted = x_col_float[sort_idx]
        y_sorted = y_int[sort_idx]

        # Identify positions where consecutive x values differ —
        # these are the only valid split positions.
        split_mask = x_sorted[:-1] != x_sorted[1:]  # shape (n-1,)
        if not np.any(split_mask):
            return None, 0.0

        valid_idx = np.where(split_mask)[0]  # split after these positions
        thresholds = (x_sorted[valid_idx] + x_sorted[valid_idx + 1]) / 2.0
        n_lefts = valid_idx + 1  # number of samples to the left

        # Original entropy H(y)
        total_counts = np.bincount(y_int, minlength=n_classes).astype(float)
        p_total = total_counts / n
        orig_ent = -np.sum(p_total[p_total > 0] * np.log2(p_total[p_total > 0] + 1e-9))

        # Cumulative class counts for the left partition at each split point
        # cum_counts[i, c] = number of class-c samples in positions 0..i
        one_hot = np.zeros((n, n_classes), dtype=float)
        one_hot[np.arange(n), y_sorted] = 1.0




        cum_counts = np.cumsum(one_hot, axis=0)  # shape (n, n_classes)

        # Left counts at each valid split: shape (k, n_classes)
        left_counts = cum_counts[valid_idx]              # cum at position valid_idx


        right_counts = total_counts[np.newaxis, :] - left_counts

        n_lefts_f = n_lefts.astype(float)[:, np.newaxis]   # (k, 1)


        n_rights_f = (n - n_lefts).astype(float)[:, np.newaxis
        ]

        # Entropy of left and right partitions (vectorised)
        def vec_entropy(counts, totals):
            # counts: (k, n_classes), totals: (k, 1)
            p = counts / np.maximum(totals, 1)
            with np.errstate(divide='ignore', invalid='ignore'):


                log_p = np.where(p > 0, np.log2(p + 1e-9), 0.0)
            return -np.sum(p * log_p, axis=1)  # shape (k,)

        ent_left  = vec_entropy(left_counts,  n_lefts_f)
        ent_right = vec_entropy(right_counts, n_rights_f)

        n_l = n_lefts.astype(float)
        n_r = (n - n_lefts).astype(float)

        weighted_ent = (n_l / n) * ent_left + (n_r / n) * ent_right


        gains = orig_ent - weighted_ent

        best_pos = int(np.argmax(gains))
        best_gain = float(gains[best_pos])

        best_threshold = float(thresholds[best_pos])

        return best_threshold, best_gain

    def best_attribute(self, X, y, attributes):
        """Find the attribute (and threshold for numerical) with the highest information gain."""
        best_gain = -1.0
        best_attr = None
        best_threshold = None

        for attr in attributes:
            if attr in self.numerical_attrs:
                x_col = X[:, attr].astype(float)

                threshold, gain = self.find_best_threshold(x_col, y)

                if threshold is None:
                    continue
            else:
                gain = self.information_gain(X[:, attr], y)
                threshold = None

            if gain > best_gain:
                best_gain = gain

                best_attr = attr
                best_threshold = threshold

        return best_attr, best_threshold, best_gain

    #- Gini impurity (kept from previous version) -----------------------------------------------------------

    def gini_impurity(self, y):
        label_counts = Counter(y)
        total = len(y)
        impurity = 1.0

        for count in label_counts.values():
            prob = count / total

            impurity -= prob ** 2
        return impurity

    #  Prediction -----------------------------------------------------------

    def predict(self, X):
        return np.array([self.predict_row(self.root, x) for x in X])

    def predict_row(self, node, x):
        if node.label is not None:
            return node.label

        if node.threshold is not None:
            # Numerical split
            val = float(x[node.attribute])



            key = '<' if val < node.threshold else '>='
            if key in node.children:


                return self.predict_row(node.children[key], x)
            # Fallback (shouldn't normally happen)
            return list(node.children.values())[0].label

        else:
            # Categorical split
            attr_value = x[node.attribute]
            if attr_value in node.children:



                return self.predict_row(node.children[attr_value], x)
            else:
                # Handle unseen categorical value at prediction time
                leaf_labels = [c.label for c in node.children.values() if c.label is not None]

                if not leaf_labels:
                    return None
                return Counter(leaf_labels).most_common(1)[0][0]
