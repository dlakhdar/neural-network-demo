import numpy as np
import pytest
import tensorflow as tf

from src.neural_network import hot_encode, prepare_data

output_length = 10


def test_preparation_data():
    x_train, y_train, x_test, y_test = prepare_data()

    # check that y_train and test are hot encoded properly
    assert len(y_train[0]) == output_length
    assert len(y_test[0]) == output_length
    assert np.count_nonzero(y_train[0]) == 1
    assert np.count_nonzero(y_test[0]) == 1

    # check shapes of each ste
    assert len(x_train) == 60000
    assert len(y_train) == 60000
    assert len(x_test) == 10000
    assert len(y_test) == 10000

    # check that there exist non-zero values
    assert np.any(x_train != 0)
    assert np.any(y_train != 0)
    assert np.any(x_test != 0)
    assert np.any(y_test != 0)
