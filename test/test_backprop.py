# Define the neural network
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from neuralnet import NeuralNetwork
from neuralnet.activations import sigmoid, derivative_sigmoid
from neuralnet.losses import mse_grad

torch.set_default_dtype(torch.float32)  # Set global default dtype to float64

inputs = [[1.0, 2.0]]
outputs = [[0.5, 0.75]]

layer_sizes = [2, 3, 4, 2]

X = np.array(inputs, dtype=np.float32)
Y = np.array(outputs, dtype=np.float32)


class SimpleNN(nn.Module):
    def __init__(self, layer_sizes):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


def mse_loss(W, a, b, y):
    return 1 / 2 * ((W @ a) + b - y) ** 2


@pytest.mark.order(5)
def loss_descent(X, Y, net, trials):
    X = np.array([[1, 2]])
    Y = np.array([[0.5, 0.75]])
    net = NeuralNetwork(layer_sizes=[2, 1, 2],
                        activation_function=sigmoid,
                        activation_derivative=derivative_sigmoid,
                        cost_function=None,
                        cost_grad=mse_grad)

    y_pred = net.predict(Y)
    initial_loss = (Y - y_pred) ** 2
    initial_loss_mean = sum(initial_loss[0]) / 2
    print(f"intial loss: {initial_loss_mean}")
    loss_mean = 0
    for _ in range(0, 5):
        net.train(X, Y, minibatch=False, iterations=1, η=5)
        y_pred = net.predict(Y)
        loss = (Y - y_pred) ** 2
        loss_mean = sum(loss[0]) / 2

    assert initial_loss_mean > loss_mean


#
@pytest.mark.order(4)
def test_backprop():
    # initialize my network
    net = NeuralNetwork(layer_sizes=layer_sizes,
                        activation_function=sigmoid,
                        activation_derivative=derivative_sigmoid,
                        cost_function=None,
                        cost_grad=mse_grad)
    w_grads = [np.zeros(matrix.shape) for matrix in net.weights]
    b_grads = [np.zeros(vector.shape) for vector in net.bias]

    # Initialize the torch_net
    torch_net = SimpleNN(layer_sizes=layer_sizes)

    with torch.no_grad():  # Disable gradient computation during assignment
        for i, layer in enumerate(torch_net.layers):
            layer.weight = nn.Parameter(torch.from_numpy(net.weights[i]).float())
            layer.bias = nn.Parameter(torch.from_numpy(net.bias[i]).float())

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(torch_net.parameters(), lr=0.01)

    # Forward pass
    torch_output = torch_net(torch.from_numpy(X))

    # Compute the loss
    loss = criterion(torch_output, torch.from_numpy(Y))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # # Retrieve gradients
    # print("Gradients:")
    # for name, param in torch_net.named_parameters():
    #     if param.grad is not None:
    #         print(f"{name} - grad:\n{param.grad}")

    for x, y in zip(X, Y):
        # print("x id:",id(x))
        z0 = net.weights[0] @ x + net.bias[0]
        zs = [z0]
        a0 = net.activation_f(z0)
        activations = [a0]
        for l in range(1, net.layer_n - 1, 1):
            # print("layers:",l,l-1)
            zl = net.weights[l] @ activations[l - 1] + net.bias[l]
            activation = net.activation_f(zl)
            # print("activation:", activation)
            zs.append(zl)
            activations.append(activation)

        z_output = zs[-1]
        a_output = activations[-1]
        output_error = net.cost_grad(a_output, y) * net.activation_df(z_output)
        errors = [output_error]
        for l in range(net.hidden_layer_n, 0, -1):
            error = net.weights[l].T @ errors[-1] * net.activation_df(zs[l - 1])
            errors.append(error)

        errors.reverse()
        # compute sum of error
        for l in range(0, net.hidden_layer_n + 1, 1):
            w_grads[l] += np.outer(errors[l], activations[l - 1] if l > 0 else x)
            # print(w_grads)
            b_grads[l] += errors[l]

    # i = 0
    # for (w_grad, b_grad) in zip(w_grads,b_grads):
    #     print(f"layer {i} - grad:\n{w_grad, b_grad}")
    #     i += 1

    torch_grads = list(torch_net.named_parameters())
    # Loop through the gradients and compare
    for i in range(len(w_grads)):
        # Compare weight gradients
        assert np.allclose(
            w_grads[i], torch_grads[2 * i][1].grad.numpy()
        ), f"Weight gradient mismatch at layer {i}"
        # Compare bias gradients
        assert np.allclose(
            b_grads[i], torch_grads[2 * i + 1][1].grad.numpy()
        ), f"Bias gradient mismatch at layer {i}"
