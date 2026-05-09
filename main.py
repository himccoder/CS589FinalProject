from encodings.punycode import digits
from typing import List
import matplotlib.pyplot as plt
from neural_net import NeuralNetwork
import numpy as np
import k_fold
from sklearn import datasets

def test_model(
        k: int, 
        dataset: str, 
        nn_layers: List[int], 
        alpha: float, 
        lambda_: float, 
        epochs: int,
        binary: bool, 
        label_encoding: bool,
        batch_size: int, 
        _label: str = "label",
        _X: np.ndarray = None, 
        _y: np.ndarray = None):
    '''
    layers does not need to include the input layer size
    '''
    k_fold_instance = k_fold.k_fold(k, label_encoding, binary)

    if dataset == None or dataset == "Digits":
        k_fold_instance.import_data(X=_X, y=_y)
    else:
        k_fold_instance.load_data(csv_path=f"./supporting_files/{dataset}.csv", target_col=_label)

    accuracies = []
    F1_scores = []

    X_train, X_test, y_train, y_test, input_dim = k_fold_instance.k_fold_split(fold_index=0)

    if nn_layers[0] != input_dim:
        # prepend input layer size to nn_layers if not already included
        nn_layers = [input_dim] + nn_layers

    print(f'Network Architecture: {nn_layers}')

    nn = NeuralNetwork(layers=nn_layers, binary=binary, alpha=alpha, lambda_=lambda_)
    y_arg = y_train.reshape(-1, 1) if binary else y_train
    nn.train(X_train, y_arg, epochs=epochs, batch_size=batch_size)

    print("Metrics on test set for fold 1:")
    if binary:
        metrics = nn.get_metrics(X_test, y_test.reshape(-1, 1))
    else:
        metrics = nn.get_metrics(X_test, y_test)

    print()
    for metric, value in metrics.items():
        if metric == "accuracy":
            accuracies.append(value)
        elif metric == "f1_score":
            F1_scores.append(value)

        print(f"{metric.capitalize()}: {value:.4f}")
    print()

    for fold_idx in range(1, k_fold_instance.get_k()):

        X_train, X_test, y_train, y_test, _ = k_fold_instance.k_fold_split(fold_index=fold_idx)

        nn = NeuralNetwork(layers=nn_layers, binary=binary, alpha=alpha, lambda_=lambda_)
        y_arg = y_train.reshape(-1, 1) if binary else y_train
        nn.train(X_train, y_arg, epochs=epochs, batch_size=batch_size)

        print(f"Metrics on test set for fold {fold_idx + 1}:")
        if binary:
            metrics = nn.get_metrics(X_test, y_test.reshape(-1, 1))
        else:
            metrics = nn.get_metrics(X_test, y_test)

        print()
        for metric, value in metrics.items():
            if metric == "accuracy":
                accuracies.append(value)
            elif metric == "f1_score":
                F1_scores.append(value)

            print(f"{metric.capitalize()}: {value:.4f}")
        print()

    print(f"Average Accuracy across {k} folds: {np.mean(accuracies):.4f}")
    print(f"Average F1 Score across {k} folds: {np.mean(F1_scores):.4f}")

def plot_learning_curves(
        k: int, 
        dataset: str, 
        nn_layers: List[int], 
        alpha: float, 
        lambda_: float, 
        epochs: int,
        binary: bool, 
        label_encoding: bool,
        batch_size: int, 
        _label: str = "label",
        _X: np.ndarray = None, 
        _y: np.ndarray = None):

    k_fold_instance = k_fold.k_fold(k, label_encoding, binary)

    if dataset is None or dataset == "Digits":
        k_fold_instance.import_data(X=_X, y=_y)
    else:
        k_fold_instance.load_data(csv_path=f"./supporting_files/{dataset}.csv", target_col=_label)

    # get input dim once
    _, _, _, _, input_dim = k_fold_instance.k_fold_split(fold_index=0)

    if nn_layers[0] != input_dim:
        nn_layers = [input_dim] + nn_layers

    all_test_losses = []

    for i in range(k):
        X_train, X_test, y_train, y_test, _ = k_fold_instance.k_fold_split(i)

        # fix label shapes
        if binary:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
        else:
            if y_train.ndim == 1:
                num_classes = len(np.unique(y_train))
                y_train = np.eye(num_classes)[y_train]
                y_test = np.eye(num_classes)[y_test]

        # New NN for each fold
        nn = NeuralNetwork(
            layers=nn_layers,
            binary=binary,
            alpha=alpha,
            lambda_=lambda_
        )

        lc = nn.learning_curve(
            X_train, y_train,
            X_test, y_test,
            epochs=epochs,
            batch_size=batch_size
        )

        all_test_losses.append(lc["test_loss_mean"])

        # store sample sizes once
        if i == 0:
            sample_sizes = lc["sample_sizes"]

    # ensure same length (important!)
    min_len = min(len(l) for l in all_test_losses)
    all_test_losses = [l[:min_len] for l in all_test_losses]
    sample_sizes = sample_sizes[:min_len]

    avg_test_losses = np.mean(all_test_losses, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, avg_test_losses, label="Test Loss")
    plt.xlabel("Instances")
    plt.ylabel("Loss")
    plt.title(f"{dataset.capitalize()} Learning Curve\nLayers: {nn_layers}, Alpha: {alpha}, Lambda: {lambda_}")
    plt.legend()
    # plt.savefig(f"./graphs/{dataset}_learning_curve.png")
    plt.show()
    plt.close()

if __name__ == "__main__":

    # --------------------------------------

    # Test different network architectures and regularization parameters

    # test a specific model on a specified dataset, only need to provide name of the dataset
    # don't need to provide input dimension in the nn_layers argument, it will be automatically prepended based on the dataset used


    # Network Architectures
    # [input_dim, 4, output_dim], reg = 0, reg = 0.1
    # [input_dim, 10, output_dim], reg = 0, reg = 0.1
    # [input_dim, 8, 4, output_dim], reg = 0, reg = 0.1
    # [input_dim, 16, 8, 8, output_dim], reg = 0, reg = 0.1


    # # # test on the parkinsons dataset
    # test_model(
    #     k=10,
    #     dataset="parkinsons",
    #     nn_layers=[16, 8, 8, 1],
    #     alpha=0.01,
    #     lambda_=0.1,
    #     epochs=500,
    #     binary=True,
    #     label_encoding=False,
    #     batch_size=64,
    #     _label="Diagnosis"
    #     )

    # # test on the credit approval dataset
    # test_model(
    #     k=10,
    #     dataset="credit_approval",
    #     nn_layers=[16, 16, 8, 1],
    #     alpha=0.1,
    #     lambda_=0.1,
    #     epochs=100,
    #     binary=True,
    #     label_encoding=False,
    #     batch_size=64
    #     )

    # # test on the rice dataset
    # test_model(
    #     k=10,
    #     dataset="rice",
    #     nn_layers=[16, 8, 8, 1],
    #     alpha=0.01,
    #     lambda_=0.1,
    #     epochs=50,
    #     binary=True,
    #     label_encoding=True,
    #     batch_size=64
    #     )


    # # test with sklearn's digit dataset
    # digits = datasets.load_digits(return_X_y=True)
    # # one hot encode the target variable
    # y_one_hot = np.eye(10)[digits[1]]
    # test_model(
    #     k=10,
    #     dataset=None,
    #     nn_layers=[8, 10, 8, 10],
    #     alpha=0.1,
    #     lambda_=0,
    #     epochs=50,
    #     binary = False,
    #     label_encoding=False,
    #     batch_size=64,
    #     _X=digits[0],
    #     _y=y_one_hot)

    # # # test with sklearn's cover type dataset
    # cov_type = datasets.fetch_covtype(return_X_y=True)
    # # # one hot encode the target variable
    # y_one_hot = np.eye(7)[cov_type[1] - 1] # shift classes from 1-7 to 0-6
    # test_model(
    #   k=10,
    #   dataset=None,
    #   nn_layers=[4, 7],
    #   alpha=0.1,
    #   lambda_=0.1,
    #   epochs=50,
    #   binary = False,
    #   label_encoding=False,
    #   batch_size=64,
    #   _X=cov_type[0],
    #   _y=y_one_hot)

    # --------------------------------------

    # # Plot learning curves for a specific model
    # plot_learning_curves(
    #     k=10,
    #     dataset="parkinsons",
    #     nn_layers=[10, 1],
    #     alpha=0.01,
    #     lambda_=0,
    #     binary=True,
    #     label_encoding=False,
    #     epochs=300,
    #     batch_size=64,
    #     _label="Diagnosis"
    # )

    # plot_learning_curves(
    #     k=10,
    #     dataset="credit_approval",
    #     nn_layers=[10, 1],
    #     alpha=0.1,
    #     lambda_=0.1,
    #     binary=True,
    #     label_encoding=False,
    #     epochs=100,
    #     batch_size=64
    # )

    # plot_learning_curves(
    #     k=10,
    #     dataset="rice",
    #     nn_layers=[10, 1],
    #     alpha=0.01,
    #     lambda_=0,
    #     binary=True,
    #     label_encoding=True,
    #     epochs=50,
    #     batch_size=64
    # )

    # digits = datasets.load_digits(return_X_y=True)
    # # one hot encode the target variable
    # y_one_hot = np.eye(10)[digits[1]]

    # print(len(digits[0]))
    # plot_learning_curves(
    #     k=10,
    #     dataset="Digits",
    #     nn_layers=[10, 10],
    #     alpha=0.1,
    #     lambda_=0.1,
    #     binary=False,
    #     label_encoding=False,
    #     epochs=50,
    #     batch_size=64,
    #     _X = digits[0],
    #     _y = y_one_hot,
    # )

    pass