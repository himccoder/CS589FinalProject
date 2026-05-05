from typing import List, Optional
import numpy as np

class NeuralNetwork:

    def __init__(self, layers: List[int], binary: bool, alpha: float = 0.5, lambda_: float = 0.0, loss_threshold: float = 1e-8):
        # layers is a list of integers where each integer represents the number of neurons in that layer
        self.layers = layers
        self.weights = []
        self.binary = binary
        self.alpha = alpha
        self.lambda_ = lambda_
        self.loss_threshold = loss_threshold

        for i in range(len(layers) - 1):
            # Rows = neurons in next layer, Cols = neurons in current layer + 1 (bias)
            weight_matrix = np.random.randn(layers[i + 1], layers[i] + 1)
            self.weights.append(weight_matrix)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def softmax(self, z):
        e = np.exp(z - np.max(z))  # subtract max for numerical stability
        return e / e.sum()

    def forward(self, X, layer):
        weight = self.weights[layer]
        z = np.dot(weight, X)
        # Use softmax on output layer for multiclass, sigmoid otherwise
        is_output = (layer == len(self.layers) - 2)
        self.activation = self.softmax(z) if (is_output and not self.binary) else self.sigmoid(z)
        return self.activation
    
    def forward_pass(self, input_data: List[List[float]]) -> List[float]:
        '''
        Forward pass through the network for a single instance of input data. This will 
        compute the activations for each layer and store them for use in backpropagation.
        '''
        all_activations = []

        for i in range(len(input_data)):
            activations = []
            activation = input_data[i]

            activation = np.insert(activation, 0, 1)  # insert bias term at the beginning
            activations.append(activation)

            for layer in range(len(self.layers) - 1):
                activation = self.forward(activation, layer)
                activation = np.insert(activation, 0, 1)  # insert bias for next layer
                activations.append(activation)
            
            # output layer will have bias term as well, but we will ignore it in the loss and metrics calculations

            all_activations.append(activations)

        return all_activations

    def backward_pass(self, all_activations: List[List[List[float]]], y_true: List[float]):
        n = len(all_activations)  # batch size
 
        # D[k] accumulates the gradient for self.weights[k] across all instances
        D = [np.zeros_like(W) for W in self.weights]
 
        for activations, y_i in zip(all_activations, y_true):
            # activations[k+1] is the output of the layer with weights self.weights[k],
            # output layer delta defined as: output - y
            # strip bias at index 0 for final output layer
            np.atleast_1d(y_i) # ensure y_i is array-like for consistent indexing
            output = activations[-1][1:]
            delta_out = output - y_i
 
            # deltas[k] will hold the delta for the layer whose weights are self.weights[k]
            deltas = [None] * len(self.weights)
            deltas[-1] = delta_out                                    # delta for last weight matrix
 
            # calculate hidden layer deltas from L-1 to 0
            for k in range(len(self.weights) - 2, -1, -1):
                # self.weights[k+1] shape: (neurons_{k+2}, neurons_{k+1} + 1)
                # Drop bias column (col 0)
                W_no_bias = self.weights[k + 1][:, 1:] # (neurons_{k+2}, neurons_{k+1})
 
                # activations[k+1] corresponds to self.weights[k]; strip bias for sigmoid'
                a = activations[k + 1][1:] # (neurons_{k+1},)
 
                # δ(k) = (W(k+1)^T · δ(k+1)) .* a .* (1 - a)
                deltas[k] = W_no_bias.T @ deltas[k + 1] * (a * (1 - a))  # (neurons_{k+1},)
 
            # accumulate gradient for each layer
            # D[k] += δ(k+1) · a(k)^T  — outer product gives correct weight-matrix shape
            # activations[k] is the input to self.weights[k], including its bias term
            # outer product multiplies delta with activation to get the gradient for all weights in that layer at once
            for k in range(len(self.weights)):
                layer_gradient = np.outer(deltas[k], activations[k])  # shape matches self.weights[k]

                # TESTING
                # print gradient for individual layer and instance
                # print("Layer", k, "gradient for instance:\n", layer_gradient, "\n")

                D[k] += layer_gradient          # shape matches self.weights[k]

            # TESTING
            # print("deltas for instance:",)
            # for i, delta in enumerate(deltas):
            #     print(f"delta[{i}]:\n{delta}\n")

        for k in range(len(self.weights)):
            P = self.lambda_ * self.weights[k].copy() # copy to avoid modifying original weights
            P[:, 0] = 0.0 # don't regularize biases
 
            # obtain average gradient
            D[k] = (1 / n) * (D[k] + P)

        # TESTING
        # print("Averaged gradients for the batch: ")
        # for i, grad in enumerate(D):
        #     print(f"Layer {i+1}:\n{grad}\n")

        # weight update for the entire batch
        for k in range(len(self.weights)):
            self.weights[k] -= self.alpha * D[k]


    def compute_loss(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        m = len(y_true)

        if self.binary:
            J = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            J = -1/m * np.sum(y_true * np.log(y_pred))  # categorical cross-entropy

        reg_penalty = 0.0
        if self.lambda_ > 0:
            for W in self.weights:
                reg_penalty += np.sum(W[:, 1:] ** 2)
            reg_penalty *= self.lambda_ / (2 * m)

        return J + reg_penalty

    def train(self, X: List[List[float]], y: List[List[float]], epochs: int, batch_size: int = None):
        '''Mini-batch gradient descent training loop.'''
        X = np.array(X)
        y = np.array(y)
 
        if batch_size is None:
            batch_size = len(X)

        prev_loss = float('inf')
 
        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
 
            J = 0.0
 
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
 
                # Forward pass for every instance in the batch
                all_activations = self.forward_pass(X_batch)
 
                # Backward pass: accumulate gradients over the batch, then update once
                self.backward_pass(all_activations, y_batch)
 
                # Collect output predictions (strip bias from final activation) for loss
                batch_preds = np.array([a[-1][1:] for a in all_activations])
 
                # Accumulate loss over the batch
                J += self.compute_loss(np.array(batch_preds), y_batch)
 
            avg_loss = J / (len(X) / batch_size)
 
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

            # alternate stopping criteria, check if loss improvement is below a threshold
            if epoch > 0 and abs(prev_loss - avg_loss) < self.loss_threshold:
                print(f"Stopping early at epoch {epoch} due to minimal loss improvement.")
                break
            prev_loss = avg_loss

    def predict(self, X: List[List[float]]) -> List[List[float]]:
        '''Predict output probabilities for the given input data.'''
        all_activations = self.forward_pass(X)
        return np.array([a[-1][1:] for a in all_activations])  # strip bias from output layer activations

    def get_metrics(self, X: List[List[float]], y_true: List[List[float]]) -> dict:
        '''Compute accuracy, precision, recall, and F1 score for the given data.'''
        y_pred = self.predict(X)

        # # TESTING
        # print("Predictions vs True Labels:")
        # for i in range(len(y_true)):
        #     print(f"Instance {i}: Predicted={y_pred[i][0]:.4f}, True={y_true[i][0]}")

        return self.compute_metrics(y_pred, y_true)

    def compute_metrics(self, y_pred: List[float], y_true: List[float]) -> dict:
        '''
        Compute accuracy and F1 score
        '''

        # implement thresholding at 0.5 for binary classification
        if self.binary:
            y_pred = (y_pred >= 0.5).astype(int)

            accuracy = np.mean(np.array(y_pred) == np.array(y_true))
        
            tp = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 1))
            fp = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 0))
            fn = np.sum((np.array(y_pred) == 0) & (np.array(y_true) == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }
        else:
            # take the largest value as the predicted class
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)
            # print(F"Predicted classes: {y_pred}")

            accuracy = np.mean(np.array(y_pred) == np.array(y_true))

            return {
                "accuracy": accuracy,
            }


    def get_weights(self):
        return self.weights
    
    def set_weights(self, weight_matrices: List[np.ndarray]):
        '''Set weights of network to custom values for testing'''
        if len(weight_matrices) != len(self.weights):
            raise ValueError("Number of weight matrices must match the number of layers - 1")
        for i in range(len(weight_matrices)):
            if weight_matrices[i].shape != self.weights[i].shape:
                raise ValueError(
                    f"Weight matrix for layer {i} has incorrect shape. "
                    f"Expected {self.weights[i].shape}, got {weight_matrices[i].shape}"
                )
            self.weights[i] = weight_matrices[i]

    def learning_curve(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_sizes: Optional[List[int]] = None,
        epochs: int = 500,
        batch_size: int = None,
        step: int = 5,
    ) -> dict:
        '''
        Train model with access to portions of the training data and evaluate on test set to generate a learning curve
        '''

        if sample_sizes is None:
            sample_sizes = list(range(step, len(X_train) + 1, step))
            # include the full training set
            if len(X_train) not in sample_sizes:
                sample_sizes.append(len(X_train))
 
        # save the initial weights so we can reset before each run
        initial_weights = [W.copy() for W in self.weights]
 
        test_losses = []
 
        for n in sample_sizes:
            # reset weights to the same initialization for a fair comparison
            self.weights = [W.copy() for W in initial_weights]
 
            X_sub = X_train[:n]
            y_sub = y_train[:n]
 
            # suppress per-epoch printing during learning curve generation
            _orig_threshold = self.loss_threshold
            self.loss_threshold = 1e-8
 
            # temporarily silence epoch prints
            import sys, io
            _stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                self.train(X_sub, y_sub, epochs=epochs, batch_size=batch_size)
            finally:
                sys.stdout = _stdout
                self.loss_threshold = _orig_threshold
 
            y_pred = self.predict(X_test)
            loss = self.compute_loss(y_pred, y_test)
            test_losses.append(loss)
            print(f"  n={n:4d} | Test J: {loss:.6f}")
 
        # restore original weights after the sweep
        self.weights = [W.copy() for W in initial_weights]
 
        return {
            "sample_sizes": sample_sizes,
            "test_losses": test_losses,
        }

    def numeric_gradient_estimation(self, X: List[List[float]], y: List[List[float]], epsilon: float = 1e-5) -> List[np.ndarray]:
        '''
        Compute numerical gradient approximation for each weight matrix in the network.
        This is used for testing the correctness of the backpropagation implementation.
        '''
        numerical_grads = []
 
        for k in range(len(self.weights)):
            grad_k = np.zeros_like(self.weights[k])
 
            for i in range(self.weights[k].shape[0]):
                for j in range(self.weights[k].shape[1]):
                    original_value = self.weights[k][i, j]
 
                    # Compute loss with positive perturbation
                    self.weights[k][i, j] = original_value + epsilon
                    loss_plus = self.compute_loss(self.predict(X), y)
 
                    # Compute loss with negative perturbation
                    self.weights[k][i, j] = original_value - epsilon
                    loss_minus = self.compute_loss(self.predict(X), y)
 
                    # Approximate gradient
                    grad_k[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
 
                    # Restore original weight value
                    self.weights[k][i, j] = original_value
 
            numerical_grads.append(grad_k)
 
        return numerical_grads