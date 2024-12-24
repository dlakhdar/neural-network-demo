import numpy as np
import pytest

from src.neural_network import NeuralNetwork


def test_initialization() -> None:
    mock_x = np.array([np.random.randint(0, 10, 3) for _ in range(3)])
    mock_y = np.array([np.random.randint(0, 10, 3) for _ in range(3)])
    net = NeuralNetwork(mock_x, mock_y, 3, [4, 3, 2])

    # check bweights:
    assert len(net.weights) == 4
    assert net.weights[0].shape == (4, 3)
    assert net.weights[1].shape == (3, 4)
    assert net.weights[2].shape == (2, 3)
    assert net.weights[3].shape == (3, 2)

    # check bias:
    assert len(net.bias) == 4
    assert len(net.bias[0]) == 4
    assert len(net.bias[1]) == 3
    assert len(net.bias[2]) == 2
    assert len(net.bias[3]) == 3
    print

    # check activation is vectorized and sensitive
    vectorized_result = net.activation_f(mock_x[0])
    assert type(vectorized_result) == np.ndarray
    assert np.all([vectorized_result, net.activation_f(mock_x[1])])
