import numpy as np
import pytest
from neuralnet import NeuralNetwork, feedforward, sigmoid


@pytest.mark.order(3)
def test_feedforward():
    mock_x = np.array([[1, 2]])
    net = NeuralNetwork([2, 2, 2, 2])

    # reassign explicitly defined weight matrices
    net.weights[0] = np.array([[0.5, 0.6], [0.8, 0.9]])
    net.weights[1] = np.array([[0.3, 0.2], [0.7, 0.8]])
    net.weights[2] = np.array([[0.1, 0.7], [0.2, 0.3]])

    # reassign explicitly defined bias matrices
    net.bias[0] = np.array([0.3, 0.4])
    net.bias[1] = np.array([0.4, 0.8])
    net.bias[2] = np.array([0.5, 0.9])

    # check feedforward result same as "hand calculated"

    activation0 = sigmoid(net.weights[0] @ mock_x[0] + net.bias[0])
    activation1 = sigmoid(net.weights[1] @ activation0 + net.bias[1])
    activation = sigmoid(net.weights[2] @ activation1 + net.bias[2])

    assert np.array_equal(
        feedforward(
            mock_x[0], net.activation_f, net.layer_n - 1, net.weights, net.bias
        ),
        activation,
    )
