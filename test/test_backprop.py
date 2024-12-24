import numpy as np
import pytest

from src.neural_network import NeuralNetwork, feedforward


def test_backprop():
    mock_x = np.array([[1, 2], [2, 3]])
    mock_y = np.array([[1, 2], [2, 3]])
    net = NeuralNetwork(mock_x, mock_y, 2, [2, 2])

    # reassign explicitly defined weight matrices
    net.weights[0] = np.array([[0.5, 0.6], [0.8, 0.9]])
    net.weights[1] = np.array([[0.3, 0.2], [0.7, 0.8]])
    net.weights[2] = np.array([[0.1, 0.7], [0.2, 0.3]])

    # reassign explicitly defined bias matrices
    net.bias[0] = np.array([0.3, 0.4])
    net.bias[1] = np.array([0.4, 0.8])
    net.bias[2] = np.array([0.5, 0.9])

    # check feedforward result same as "hand calculated"

    output = feedforward(
        mock_x[0], net.activation_f, net.layer_n - 1, net.weights, net.bias
    )
    print(output)

    output_error = (output - mock_y[0]) * net.activation_df(output)
    print(output_error)
