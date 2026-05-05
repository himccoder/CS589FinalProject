import os
from pathlib import Path
from random import seed
from matplotlib.pylab import seed
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class k_fold:
    
    def __init__(self, k: int):
        self.k = k
        self.X = None
        self.df = None
        self.y = None
        self.fold = None
    
    def load_data(self, csv_path: Path, target_col: str = "label"):
        print(f"Loading data from {os.path.basename(csv_path)} ...\n")
        self.df = pd.read_csv(csv_path)
        self.X = self.df.drop(columns=[target_col])
        self.y = self.df[target_col]

    def k_fold_split(self, fold_index: Optional[int] = None):
        # one hot encode before splitting to avoid fold variation

        X = self.X.copy()

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        if len(categorical_cols) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_cat = encoder.fit_transform(X[categorical_cols])
            feature_names = encoder.get_feature_names_out(categorical_cols)
            X = pd.concat([
                X.drop(columns=categorical_cols),
                pd.DataFrame(X_cat, index=X.index, columns=feature_names)
            ], axis=1)


        # stratified split the data into k folds
        '''
        '''

        class_0_idx = np.where(self.y == 0)[0]
        class_1_idx = np.where(self.y == 1)[0]

        np.random.shuffle(class_0_idx)
        np.random.shuffle(class_1_idx)

        class0_folds = np.array_split(class_0_idx, self.k)
        class1_folds = np.array_split(class_1_idx, self.k)

        self.folds = []
        for i in range(self.k):
            fold_idx = np.concatenate([class0_folds[i], class1_folds[i]])
            np.random.shuffle(fold_idx)
            self.folds.append(fold_idx)

        test_idx = self.folds[fold_index]
        train_idx = np.concatenate([self.folds[i] for i in range(self.k) if i != fold_index])

        '''
        '''

        X_train, y_train = X.iloc[train_idx].copy(), self.y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx].copy(), self.y.iloc[test_idx]

        numerical_cols = X_train.select_dtypes(include=["number"]).columns

        print(f"{len(train_idx)}")

        # scale numerical features 
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        # return numpy arrays instead of pandas dataframes/series for easier indexing during training
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values

        # return input layer dimension as well for easier neural network initialization
        return X_train, X_test, y_train, y_test, X_train.shape[1]

    def get_k(self):
        return self.k