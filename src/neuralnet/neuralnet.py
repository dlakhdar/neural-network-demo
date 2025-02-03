import numpy as np
from numba import njit, int32, int64, float32, float64
from numba.types import ListType, FunctionType
from neuralnet.activations import sigmoid, derivative_sigmoid
from neuralnet.losses import mse, mse_grad

# TODO: produce help strings
# TODO: better type annotation
# TODO: implement different initializations
# TODO: implement AD with jax
# TODO: eliminate use of lists , optimize with jax and numba
# TODO: implement adam
# TODO: add a seed option
# TODO:
# TODO: add grad method , returns grad of neural network

fire = 1
output_length = 10


@njit
def calculate_validation_rate(predicted_y: list[np.ndarray],
                              y: list[np.ndarray]) -> float | int:
    """
    Calculate the validation rate (accuracy) for predicted and actual labels.

    Parameters:
    - predicted_y: array-like, predicted probabilities or logits (e.g., from a neural network).
    - y: array-like, one-hot encoded true labels.

    Returns:
    - float, the accuracy rate as the proportion of correctly predicted samples.
    """
    predicted_indices = np.array(list(map(np.argmax, predicted_y)))
    true_indices = np.array(list(map(np.argmax, y)))
    correct_predictions = np.sum(predicted_indices == true_indices)
    accuracy = correct_predictions / len(y)
    return accuracy


def feedforward(
    x: np.ndarray,
    W: list[np.ndarray],
    b: list[np.ndarray],
    σ: callable,
) -> np.ndarray:
    """
    Perform a feedforward computation in a neural network.

    Parameters
    ----------
    x : np.array
        The input array to the neural network. Typically a vector or batch of vectors.
    σ : callable
        The activation function applied element-wise at each layer
          (e.g., ReLU, sigmoid, or tanh).
    n : int | float
        The number of layers in the neural network. Assumes layers are indexed from 0 to n-1.
    W : NeuralNetwork.weights
        A list or array of weight matrices, where `W[l]` is the weight matrix for layer `l`.
        Each matrix should have dimensions suitable for the connections between layers.
    b : NeuralNetwork.bias
        A list or array of bias vectors, where `b[l]` is the bias vector for layer `l`.
        Each vector should have dimensions matching the output size of the respective layer.

    Returns
    -------
    np.array
        The output of the neural network after applying all layers.

    Notes
    -----
    - The function assumes that the number of layers (`n`) matches the length of `W` and `b`.
    - The activation function (`σ`) is applied after the affine transformation at each layer:
      `activation = σ(W[l] @ activation + b[l])`.
    - `x` is treated as the initial activation for layer 0.

    Example
    -------
    >>> import numpy as np
    >>> def relu(x):
    ...     return np.maximum(0, x)
    ...
    >>> x = np.array([1, 2])
    >>> W = [np.array([[0.1, 0.2], [0.3, 0.4]]),
      np.array([[0.5, 0.6]])]
    >>> b = [np.array([0.1, 0.2]), np.array([0.3])]
    >>> feedforward(x, relu, 2, W, b)
    array([0.77])  # Example output
    """

    activation = σ(W[0] @ x + b[0])
    for l in range(1, len(W)):
        activation = σ(W[l] @ activation + b[l])

    return activation


# @njit
def backpropagation(x: np.ndarray[:],
                    y: np.ndarray[:], 
                    weights: list[np.ndarray[:,:]], 
                    biases: list[np.ndarray[:]], 
                    activation_f: callable, 
                    activation_df: callable, 
                    cost_grad: callable):
    """
    Performs backpropagation for a single training example in a neural network.

    Backpropagation calculates the gradients of the weights and biases with respect
    to the loss function, enabling optimization of the network during training.

    Parameters
    ----------
    x : np.ndarray
        The input vector for the training example.
    y : np.ndarray
        The target output vector for the training example.
    weights : list of np.ndarray
        A list of weight matrices for each layer in the neural network.
        Each matrix has dimensions (neurons in current layer, neurons in previous layer).
    biases : list of np.ndarray
        A list of bias vectors for each layer in the neural network.
        Each vector has dimensions (neurons in current layer, ).
    activation_f : callable
        The activation function to apply at each layer (e.g., sigmoid, ReLU).
    activation_df : callable
        The derivative of the activation function used for backpropagation.
    cost_grad : callable
        The gradient of the cost function with respect to the activations at the output layer.

    Returns
    -------
    w_grads : list of np.ndarray
        Gradients of the weight matrices for each layer. Each matrix matches the dimensions
        of the corresponding weight matrix in `weights`.
    b_grads : list of np.ndarray
        Gradients of the bias vectors for each layer. Each vector matches the dimensions
        of the corresponding bias vector in `biases`.

    Notes
    -----
    - Forward propagation is performed to compute the activations and pre-activations (z-values)
      for all layers.
    - Backward propagation uses these values to compute errors at each layer.
    - Gradients for weights and biases are computed using these errors and activations.

    Example
    -------
    >>> import numpy as np
    >>> from numba import njit
    >>> def sigmoid(z):
    ...     return 1 / (1 + np.exp(-z))
    ...
    >>> def derivative_sigmoid(z):
    ...     sig = sigmoid(z)
    ...     return sig * (1 - sig)
    ...
    >>> x = np.array([1, 0.5])
    >>> y = np.array([0])
    >>> weights = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5, 0.6]])]
    >>> biases = [np.array([0.1, 0.2]), np.array([0.3])]
    >>> w_grads, b_grads = backpropagation(
    ...     x, y, weights, biases, sigmoid, derivative_sigmoid, lambda a, y: a - y
    ... )
    >>> w_grads
    [array([...]), array([...])]
    >>> b_grads
    [array([...]), array([...])]
    """
    zs = []
    activations = [x]

    # Forward pass
    for l in range(len(weights)):
        z = weights[l] @ activations[-1] + biases[l]
        zs.append(z)
        activations.append(activation_f(z))

    # Backward pass
    errors = [cost_grad(activations[-1], y) * activation_df(zs[-1])]
    for l in range(len(weights) - 1, 0, -1):
        errors.append(weights[l].T @ errors[-1] * activation_df(zs[l - 1]))
    errors.reverse()

    # Gradient computation
    w_grads = [np.outer(errors[l], activations[l]) for l in range(len(weights))]
    b_grads = [errors[l] for l in range(len(biases))]

    return w_grads, b_grads


class NeuralNetwork:
    """
    A class for constructing and training a fully connected neural network.

    Attributes:
        hidden_layer_n (int): Number of hidden layers in the network.
        layer_n (int): Total number of layers (input + hidden + output).
        layer_sizes (np.array): List of sizes for each layer in the network.
        bias (list): List of bias vectors for each layer.
        weights (list): List of weight matrices connecting the layers.
        activation_f (callable): Activation function (default: sigmoid).
        activation_df (callable): Derivative of the activation function.
        cost_function (callable): Cost function for training (if provided).

    Methods:
        train(minibatch=True, minibatch_pool=10, iterations=100, η=1e-6) -> 'NeuralNetwork':
            Trains the neural network using gradient descent.

    Parameters:
        hidden_layer (int): Number of hidden layers in the network.
        layer_sizes (list[int | float]): List of hidden layer sizes (default: [10]).
        activation_function (callable): Activation function for all layers (default: sigmoid).
        activation_derivative (callable): Derivative of the activation function (default: derivative_sigmoid).
        cost (callable): Cost function to minimize during training (optional).
        cost_grad (callable): Cost function gradient with respect to activations solely

    Train Method Parameters:
        minibatch (bool): Whether to use mini-batch gradient descent (default: True).
        minibatch_pool (int | float): Number of samples per mini-batch (default: 10).
        iterations (int | float): Number of training iterations (default: 100).
        η (int | float): Learning rate for gradient descent (default: 1e-6).

    Returns:
        NeuralNetwork: The trained neural network object.

    Example:
        nn = NeuralNetwork(
                           layer_sizes=[10,64, 32,1],
                           activation_function=sigmoid,
                           activation_derivative=derivative_sigmoid)
        nn.train(minibatch=True, minibatch_pool=32, iterations=1000, η=0.01)
    """

    def __init__(
        self,
        layer_sizes: (list[int] | list[float]),
        activation_function: callable,
        activation_derivative: callable,
        cost_function: callable,
        cost_grad: callable,
    ) -> "NeuralNetwork":
        self.layer_sizes = layer_sizes
        self.layer_n = len(self.layer_sizes)
        self.hidden_layer_n = len(self.layer_sizes) - 2
        self.bias = [
            np.random.randn(self.layer_sizes[i])
            for i in range(1, self.hidden_layer_n + 2)
        ]

        self.weights = [
            np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1])
            * np.sqrt(1 / self.layer_sizes[i - 1])
            for i in range(1, self.hidden_layer_n + 2)
        ]

        self.activation_f = activation_function
        self.activation_df = activation_derivative
        self.cost = cost_function
        self.cost_grad = cost_grad

    def __str__(self):
        """
        Returns a string representation of the neural network's architecture, weights, and biases.
        """
        display_str = "Neural Network Structure:\n"
        display_str += f"Number of Layers: {self.layer_n}\n"
        display_str += f"Hidden Layers: {self.hidden_layer_n}\n"
        display_str += "Layer Sizes: " + " -> ".join(map(str, self.layer_sizes)) + "\n\n"

        display_str += "Biases:\n"
        for i, bias in enumerate(self.bias, start=1):
            display_str += f"  Layer {i + 1}: Shape {bias.shape}\n"

        display_str += "\nWeights:\n"
        for i, weight in enumerate(self.weights, start=1):
            display_str += f"  Layer {i}: Shape {weight.shape}\n"

        display_str += f"\nActivation Function: {self.activation_f.__name__}\n"
        display_str += f"Activation Derivative: {self.activation_df.__name__}\n"
        display_str += f"Cost Function: {self.cost.__name__}\n"
        display_str += f"Cost Gradient: {self.cost_grad.__name__}\n"

        return display_str

    def train(
        self,
        input,
        output,
        momentum: (int | float) = None,
        minibatch: bool = True,
        minibatch_pool: (int | float) = 10,
        iterations: (int | float) = 100,
        η: (int | float) = 1e-6,
    ) -> None:
        """
        Trains the neural network using gradient descent.

        Parameters:
            minibatch (bool): Whether to use mini-batch gradient descent (default: True).
            minibatch_pool (int | float): Size of the mini-batch for training (default: 10).
            iterations (int | float): Number of training iterations (default: 100).
            η (int | float): Learning rate for gradient descent (default: 1e-6).

        Returns:
            NeuralNetwork: The trained neural network object.

        Description:
            - Implements forward propagation for each input to compute activations.
            - Performs backpropagation to compute gradients for weights and biases.
            - Updates weights and biases using gradient descent.
            - Supports mini-batch gradient descent if `minibatch` is set to True.

        Example:
            nn.train(minibatch=True, minibatch_pool=32, iterations=1000, η=0.01)
        """

        for _ in range(iterations):
            if minibatch:
                indexes = np.random.choice(input.shape[0], size=minibatch_pool)
                X, Y = input[indexes], output[indexes]
            else:
                X, Y = input, output

            w_grads = [np.zeros(matrix.shape) for matrix in self.weights]

            b_grads = [np.zeros(vector.shape) for vector in self.bias]

            if momentum is not None:
                v = [np.zeros(w.shape) for w in self.weights]

            # iterate for each set of x and y
            # find zs and as (pre-act and activation)
            for x, y in zip(X, Y):
                # print("x id:",id(x))

                # def feedforward
                z0 = self.weights[0] @ x + self.bias[0]
                zs = [z0]
                a0 = self.activation_f(z0)
                activations = [a0]
                for l in range(1, self.layer_n - 1, 1):
                    # print("layers:",l,l-1)
                    zl = self.weights[l] @ activations[l - 1] + self.bias[l]
                    activation = self.activation_f(zl)
                    # print("activation:", activation)
                    zs.append(zl)
                    activations.append(activation)

                z_output = zs[-1]
                a_output = activations[-1]
                output_error = self.cost_grad(a_output, y) * self.activation_df(
                    z_output
                )
                errors = [output_error]
                for l in range(self.hidden_layer_n, 0, -1):
                    error = (
                        self.weights[l].T @ errors[-1] * self.activation_df(zs[l - 1])
                    )
                    errors.append(error)

                errors.reverse()
                # compute sum of error
                for l in range(0, self.hidden_layer_n + 1, 1):
                    # w_grads[l] += errors[l]@activations[l].T
                    w_grads[l] += np.outer(
                        errors[l], activations[l - 1] if l > 0 else x
                    )
                    # print(w_grads)
                    b_grads[l] += errors[l]
                    # print(b_grads)

            # gradient descent
            if momentum is None:
                for l in range(0, self.hidden_layer_n + 1):
                    self.weights[l] -= η / minibatch_pool * w_grads[l]
                    self.bias[l] -= η / minibatch_pool * b_grads[l]
            else:
                γ = momentum
                for l in range(0, self.hidden_layer_n + 1):
                    v[l] = γ * v[l] + η / minibatch_pool * w_grads[l]
                    self.weights[l] -= v[l]
                    self.bias[l] -= η / minibatch_pool * b_grads[l]

    def predict(self, input: list[np.ndarray]) -> list[np.ndarray]:
        """
        Predicts the output for a given input using the trained neural network.

        Parameters:
            input (np.array): Input data to predict, where each row corresponds to a single input instance.

        Returns:
            list: A list of predictions where each prediction corresponds to the output of the neural network
                  for the corresponding input instance.

        Description:
            - Performs forward propagation through the network to compute the output layer activations.
            - Returns the final layer activations as predictions.

        Example:
            predictions = nn.predict(x_test)
        """

        results = []
        for x in input:
            # print(x,"\n")
            z0 = self.activation_f(self.weights[0] @ x + self.bias[0])
            activations = [z0]
            for l in range(1, self.layer_n - 1):
                zl = self.weights[l] @ activations[l - 1] + self.bias[l]
                a = self.activation_f(zl)
                activations.append(a)

            results.append(activations[-1])

        return results


class NeuralNetworkwMemory(NeuralNetwork):
    
    def train(self,
              input, 
              output, 
              momentum=None, 
              minibatch: bool = True,
              minibatch_pool : (int | float) = 10,
              iterations: (int | float) = 100,
              η: (int | float) = 1e-6) -> list[np.array]:
        """
        Trains the neural network using gradient descent.

        Parameters:
            minibatch (bool): Whether to use mini-batch gradient descent (default: True).
            minibatch_pool (int | float): Size of the mini-batch for training (default: 10).
            iterations (int | float): Number of training iterations (default: 100).
            η (int | float): Learning rate for gradient descent (default: 1e-6).

        Returns:
            NeuralNetwork: The trained neural network object.

        Description:
            - Implements forward propagation for each input to compute activations.
            - Performs backpropagation to compute gradients for weights and biases.
            - Updates weights and biases using gradient descent.
            - Supports mini-batch gradient descent if `minibatch` is set to True.

        Example:
            nn.train(minibatch=True, minibatch_pool=32, iterations=1000, η=0.01)
        """
        
        weight_history = []

        for _ in range(iterations):

            if minibatch:
                indexes = np.random.choice(input.shape[0], size=minibatch_pool)
                X, Y = input[indexes], output[indexes]
            else: 
                X, Y = input, output


            if momentum is not None:
                v = [ np.zeros(w.shape) for w in self.weights ]
                
            a_errors = [np.zeros(matrix.shape) for matrix in self.weights]

            b_errors = [np.zeros(vector.shape) for vector in self.bias]

            # iterate for each set of x and y
            # find zs and as (pre-act and activation)
            for x, y in zip(X, Y): 
                # print("x id:",id(x))
                z0 = self.weights[0] @ x + self.bias[0]
                zs = [z0]
                a0 = self.activation_f(z0)
                activations = [a0]
                for l in range(1, self.layer_n-1,1):
                    # print("layers:",l,l-1)
                    zl = self.weights[l] @ activations[l-1] + self.bias[l]
                    activation = self.activation_f(zl)
                    # print("activation:", activation)
                    zs.append(zl)
                    activations.append(activation)  

                z_output = zs[-1]
                a_output = activations[-1]
                output_error = self.cost_grad(a_output,y) * self.activation_df(z_output)
                errors = [output_error]
                for l in range(self.hidden_layer_n, 0, -1):
                    error = self.weights[l].T @ errors[-1] * self.activation_df(zs[l-1])
                    errors.append(error)

                errors.reverse()
                # compute sum of error
                for l in range(0,self.hidden_layer_n+1,1):
                    # a_errors[l] += errors[l]@activations[l].T
                    a_errors[l] += np.outer(errors[l], activations[l-1]  if l > 0 else x)
                    # print(a_errors)
                    b_errors[l] += errors[l] 
                        # print(b_errors)


            # gradient descent 
            if momentum == None: 
                for l in range(0, self.hidden_layer_n+1):
                    self.weights[l] -= η / minibatch_pool * a_errors[l]
                    self.bias[l] -= η / minibatch_pool * b_errors[l]
            else: 
                γ = momentum 
                for l in range(0, self.hidden_layer_n+1):
                    v[l] = γ * v[l] + η / minibatch_pool * a_errors[l]
                    self.weights[l] -= v[l]
                    self.bias[l] -= η / minibatch_pool * b_errors[l]
            

            weight_history.append(self.weights)

        return weight_history

def main():
    return None
