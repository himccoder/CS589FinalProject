# RandomForestExperiment.py
# CS 589 Final Project - Himnish
# Runs Random Forest on 4 required datasets + 1 extra credit dataset (Covertype).
# Outputs: hyperparameter table + learning curve figure per dataset.

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # save figures without needing a display
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import datasets as sk_datasets   # only used for loading data, not ML

from RandomForest import RandomForest

# Where to save figures and tables
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# CSV files live one directory up
DATA_DIR = os.path.join(os.path.dirname(__file__), '..')


# ==============================================================
# DATA LOADERS
# ==============================================================

def load_csv(path):
    """
    Load a CSV that uses the attr_num / attr_cat / label header convention.
    Returns X (object array), y (str array), numerical_attrs (list of col indices).
    """
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')

    header = lines[0].strip().split(',')
    rows   = [line.strip().split(',') for line in lines[1:] if line.strip()]
    data   = np.array(rows, dtype=object)

    label_col  = header.index('label')
    feat_cols  = [i for i in range(len(header)) if i != label_col]
    feat_names = [header[i] for i in feat_cols]

    X = data[:, feat_cols]
    y = data[:, label_col]

    # columns whose name contains '_num' are numerical
    numerical_attrs = [j for j, name in enumerate(feat_names) if '_num' in name]

    # convert numerical columns to float, fill missing with column mean
    for j in numerical_attrs:
        vals = []
        for v in X[:, j]:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(np.nan)
        col = np.array(vals, dtype=float)
        col = np.where(np.isnan(col), np.nanmean(col), col)
        X[:, j] = col

    # drop rows with empty categorical cells
    cat_attrs = [j for j in range(X.shape[1]) if j not in numerical_attrs]
    valid = [i for i in range(len(X))
             if all(str(X[i, j]).strip() != '' for j in cat_attrs)]
    X = X[valid]
    y = y[valid]

    return X, y, numerical_attrs


def load_parkinsons(path):
    """
    Load the Parkinson's CSV.
    All columns except 'Diagnosis' are numerical features.
    """
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')

    header = lines[0].strip().split(',')
    rows   = [line.strip().split(',') for line in lines[1:] if line.strip()]
    data   = np.array(rows, dtype=object)

    label_col = header.index('Diagnosis')
    feat_cols = [i for i in range(len(header)) if i != label_col]

    X = data[:, feat_cols].astype(float)
    y = data[:, label_col]
    numerical_attrs = list(range(X.shape[1]))

    return X, y, numerical_attrs


def load_digits():
    """
    Load the sklearn hand-written digits dataset.
    10 classes (0-9), 64 numerical pixel features, 1797 instances.
    """
    digits = sk_datasets.load_digits()
    X = digits.data.astype(float)
    y = digits.target.astype(str)
    numerical_attrs = list(range(X.shape[1]))
    return X, y, numerical_attrs


def load_covtype(n_samples=5000, random_state=42):
    """
    Load the Forest Covertype dataset and stratified-subsample to n_samples rows.
    Full dataset has 581k rows - we subsample to keep runtime reasonable.
    All 54 features treated as numerical.
    """
    print('  Fetching Covertype dataset (may download on first run)...')
    covtype = sk_datasets.fetch_covtype()
    X_full = covtype.data.astype(float)
    y_full = covtype.target.astype(str)

    # stratified subsample: equal share per class
    rng = np.random.RandomState(random_state)
    classes  = np.unique(y_full)
    per_class = n_samples // len(classes)
    selected = []
    for c in classes:
        idx    = np.where(y_full == c)[0]
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        selected.extend(chosen.tolist())
    rng.shuffle(selected)

    X = X_full[selected]
    y = y_full[selected]
    numerical_attrs = list(range(X.shape[1]))
    return X, y, numerical_attrs


# ==============================================================
# STRATIFIED K-FOLD
# ==============================================================

def stratified_kfold(X, y, k=10, random_state=42):
    """
    Split data into k stratified folds.
    Returns list of (X_train, y_train, X_test, y_test) tuples.
    """
    rng = np.random.RandomState(random_state)
    y_str = np.array([str(v) for v in y])

    # bucket indices by class label
    class_idx = {}
    for i, label in enumerate(y_str):
        class_idx.setdefault(label, []).append(i)

    # shuffle each bucket then cut into k chunks
    class_folds = {}
    for label, idx in class_idx.items():
        arr = np.array(idx)
        rng.shuffle(arr)
        class_folds[label] = np.array_split(arr, k)

    splits = []
    for i in range(k):
        test_idx  = np.concatenate([class_folds[c][i] for c in class_folds])
        train_idx = np.concatenate(
            [class_folds[c][j] for c in class_folds for j in range(k) if j != i]
        )
        splits.append((X[train_idx], y[train_idx], X[test_idx], y[test_idx]))

    return splits


# ==============================================================
# METRICS  (works for binary and multiclass)
# ==============================================================

def compute_metrics(y_true, y_pred):
    """
    Returns (accuracy, macro_F1).
    Macro F1 = average of per-class F1 scores - handles any number of classes.
    """
    y_true = np.array([str(v) for v in y_true])
    y_pred = np.array([str(v) for v in y_pred])

    accuracy = float(np.mean(y_true == y_pred))

    f1_per_class = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = (2 * prec * recall / (prec + recall)
                  if (prec + recall) > 0 else 0.0)
        f1_per_class.append(f1)

    macro_f1 = float(np.mean(f1_per_class))
    return accuracy, macro_f1


# ==============================================================
# CROSS-VALIDATION HELPER
# ==============================================================

def cross_validate(X, y, numerical_attrs, rf_params, k=10, random_state=42):
    """Run k-fold CV and return (mean_accuracy, mean_F1)."""
    splits = stratified_kfold(X, y, k=k, random_state=random_state)
    accs, f1s = [], []
    for X_train, y_train, X_test, y_test in splits:
        rf = RandomForest(**rf_params)
        rf.fit(X_train, y_train, numerical_attrs=numerical_attrs)
        y_pred = rf.predict(X_test)
        acc, f1 = compute_metrics(y_test, y_pred)
        accs.append(acc)
        f1s.append(f1)
    return float(np.mean(accs)), float(np.mean(f1s))


# ==============================================================
# HYPERPARAMETER GRID  (>= 6 settings as required)
# ==============================================================

def run_hyperparam_grid(name, X, y, numerical_attrs, k=10):
    """
    Grid: ntree in {5, 10, 20, 50} x max_depth in {5, 10, None} = 12 combos.
    Prints a table and returns the best (ntree, max_depth) by accuracy.
    """
    ntree_vals = [5, 10, 20, 50]
    depth_vals = [5, 10, None]   # None means grow until pure / min_samples

    print(f"\n{'='*62}")
    print(f"  {name}  -  Hyperparameter Grid  (k={k}-fold CV)")
    print(f"{'='*62}")
    print(f"  {'ntree':>6}  {'max_depth':>10}  {'Accuracy':>9}  {'F1':>7}")
    print(f"  {'-'*38}")

    best_acc, best_f1, best_params = -1.0, -1.0, None

    for ntree in ntree_vals:
        for max_depth in depth_vals:
            rf_params = dict(
                n_trees=ntree,
                max_depth=max_depth,
                min_samples_split=5,
                min_gain=1e-4
            )
            acc, f1 = cross_validate(X, y, numerical_attrs, rf_params, k=k)

            depth_str = str(max_depth) if max_depth is not None else 'None'
            print(f"  {ntree:>6}  {depth_str:>10}  {acc:>9.4f}  {f1:>7.4f}")

            if acc > best_acc:
                best_acc, best_f1 = acc, f1
                best_params = (ntree, max_depth)

    print(f"\n  >>> Best: ntree={best_params[0]}, max_depth={best_params[1]}"
          f"  =>  Acc={best_acc:.4f}, F1={best_f1:.4f}")

    return best_params, best_acc, best_f1


# ==============================================================
# LEARNING CURVE  (accuracy & F1 vs number of trees)
# ==============================================================

def run_learning_curve(name, X, y, numerical_attrs, best_max_depth, k=10):
    """
    Sweep ntree with the best max_depth from the grid search.
    Saves a figure to results/.
    """
    ntree_vals = [1, 5, 10, 20, 30, 50, 75, 100]

    print(f"\n  Learning curve  (max_depth={best_max_depth}, k={k})...")
    accs, f1s = [], []
    for ntree in ntree_vals:
        rf_params = dict(
            n_trees=ntree,
            max_depth=best_max_depth,
            min_samples_split=5,
            min_gain=1e-4
        )
        acc, f1 = cross_validate(X, y, numerical_attrs, rf_params, k=k)
        accs.append(acc)
        f1s.append(f1)
        print(f"    ntree={ntree:>3}  Acc={acc:.4f}  F1={f1:.4f}")

    # plot both metrics on one figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ntree_vals, accs, marker='o', color='steelblue',   label='Accuracy',  linewidth=2)
    ax.plot(ntree_vals, f1s,  marker='s', color='darkorange',  label='Macro F1',  linewidth=2)
    ax.set_xlabel('Number of Trees ($n_{tree}$)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'RF Performance vs Number of Trees\n({name})', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ntree_vals)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    safe = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    fpath = os.path.join(RESULTS_DIR, f'{safe}_learning_curve.png')
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved figure: {fpath}")

    return ntree_vals, accs, f1s


# ==============================================================
# MAIN
# ==============================================================

if __name__ == '__main__':
    np.random.seed(42)

    # -----------------------------------------------------------
    # Dataset definitions
    # key: short id used to pick the right loader
    # path: path to CSV (None for sklearn-loaded datasets)
    # k: folds to use (5 for Covertype to keep runtime reasonable)
    # -----------------------------------------------------------
    datasets = [
        {
            'name': 'Digits',
            'key':  'digits',
            'path':  None,
            'k':     10,
        },
        {
            'name': 'Parkinsons',
            'key':  'parkinsons',
            'path':  os.path.join(DATA_DIR, 'parkinsons.csv'),
            'k':     10,
        },
        {
            'name': 'Rice',
            'key':  'csv',
            'path':  os.path.join(DATA_DIR, 'rice.csv'),
            'k':     10,
        },
        {
            'name': 'Credit Approval',
            'key':  'csv',
            'path':  os.path.join(DATA_DIR, 'credit_approval.csv'),
            'k':     10,
        },
        {
            'name': 'Covertype (Extra Credit)',
            'key':  'covtype',
            'path':  None,
            'k':     5,    # fewer folds - dataset is larger even after subsampling
        },
    ]

    summary = {}   # name -> (best_acc, best_f1)

    for ds in datasets:
        name = ds['name']
        k    = ds['k']

        print(f"\n{'#'*62}")
        print(f"#  Dataset: {name}")
        print(f"{'#'*62}")

        # --- load ---
        if ds['key'] == 'digits':
            X, y, num_attrs = load_digits()
        elif ds['key'] == 'parkinsons':
            X, y, num_attrs = load_parkinsons(ds['path'])
        elif ds['key'] == 'csv':
            X, y, num_attrs = load_csv(ds['path'])
        else:  # covtype
            X, y, num_attrs = load_covtype(n_samples=5000)

        print(f"  Instances: {X.shape[0]}, Features: {X.shape[1]}, "
              f"Numerical: {len(num_attrs)}, Categorical: {X.shape[1]-len(num_attrs)}")
        print(f"  Classes: {sorted(set(str(c) for c in y))}")
        print(f"  Distribution: {dict(Counter(str(c) for c in y))}")

        # --- hyperparameter grid ---
        best_params, best_acc, best_f1 = run_hyperparam_grid(
            name, X, y, num_attrs, k=k
        )
        best_ntree, best_depth = best_params

        # --- learning curve ---
        run_learning_curve(name, X, y, num_attrs, best_depth, k=k)

        summary[name] = (best_acc, best_f1)

    # -----------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------
    print('\n\n' + '=' * 62)
    print('FINAL SUMMARY  -  Best hyperparameters per dataset')
    print('=' * 62)
    print(f"  {'Dataset':<28}  {'Accuracy':>9}  {'Macro F1':>9}")
    print(f"  {'-'*50}")
    for name, (acc, f1) in summary.items():
        print(f"  {name:<28}  {acc:>9.4f}  {f1:>9.4f}")
    print()
