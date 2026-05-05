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
        batch_size: int, 
        _label: str = "label",
        _X: np.ndarray = None, 
        _y: np.ndarray = None):
    '''
    layers does not need to include the input layer size
    '''
    k_fold_instance = k_fold.k_fold(k)

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

    nn = NeuralNetwork(layers=nn_layers, alpha=alpha, lambda_=lambda_)
    nn.train(X_train, y_train.reshape(-1, 1), epochs=epochs, batch_size=batch_size)

    print("Metrics on test set for fold 1:")
    metrics = nn.get_metrics(X_test, y_test.reshape(-1, 1))

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

        nn = NeuralNetwork(layers=nn_layers, alpha=alpha, lambda_=lambda_)
        nn.train(X_train, y_train.reshape(-1, 1), epochs=epochs, batch_size=batch_size)

        print(f"Metrics on test set for fold {fold_idx + 1}:")
        metrics = nn.get_metrics(X_test, y_test.reshape(-1, 1))

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

def plot_learning_curves(k: int, dataset: str, nn_layers: List[int], alpha: float, lambda_: float, epochs: int, batch_size: int):
    k_fold_instance = k_fold.k_fold(k)
    k_fold_instance.load_data(csv_path=f"./../datasets/{dataset}.csv", target_col="label")

    # select a random fold to plot the learning curve for
    X_train, X_test, y_train, y_test, input_dim = k_fold_instance.k_fold_split(fold_index=np.random.randint(0, k))

    if nn_layers[0] != input_dim:
        # prepend input layer size to nn_layers if not already included
        nn_layers = [input_dim] + nn_layers

    nn = NeuralNetwork(layers=nn_layers, alpha=alpha, lambda_=lambda_)
    learning_curve = nn.learning_curve(X_train, y_train.reshape(-1, 1), X_test, y_test.reshape(-1, 1), epochs=epochs, batch_size=batch_size)

    # plot learning curve, instances on x-axis and loss on y-axis
    plt.figure(figsize=(10, 6))
    plt.plot(learning_curve["sample_sizes"], learning_curve["test_losses"], label="Test Loss")
    plt.xlabel("Instances")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve for {dataset} Dataset\nLayers: {nn_layers}, Alpha: {alpha}, Lambda: {lambda_}")

    plt.show()

if __name__ == "__main__":

    # --------------------------------------

    # Test different network architectures and regularization parameters 

    # test a specific model on a specified dataset, only need to provide name of the dataset
    # don't need to provide input dimension in the nn_layers argument, it will be automatically prepended based on the dataset used
    
    # test_model(k=10, dataset="parkinsons", nn_layers=[4, 1], alpha=0.1, lambda_=0, epochs=50, batch_size=64, _label="Diagnosis")
    # test_model(k=10, dataset="credit_approval", nn_layers=[4, 1], alpha=0.1, lambda_=0, epochs=50, batch_size=64)
    # test_model(k=10, dataset="rice", nn_layers=[4, 1], alpha=0.1, lambda_=0, epochs=50, batch_size=64)

    # test with sklearn's digit dataset
    digits = datasets.load_digits()
    test_model(k=10, dataset=None, nn_layers=[4, 1], alpha=0.1, lambda_=0, epochs=50, batch_size=64, _X=digits.data, _y=digits.target)

    # --------------------------------------

    # Plot learning curves for a specific model
    # plot_learning_curves(k=5, dataset="wdbc", nn_layers=[6, 6, 1], alpha=0.1, lambda_=0.25, epochs=500, batch_size=64)