
import pytest 
import numpy as np 


def calculate_hamiltonian(J, S) -> (int | float):

    """
    Performs a grid search over hyperparameters for training a neural network
    with memory, evaluating combinations of learning rates, momentums, pooling sizes,
    and activation functions.

    Parameters:
    ----------
    J : list
        A list of learning rates (η) to test. Default is [1e-4, 1e-2, 1e-1].
    S : list
        A list of momentum values (γ) to test, including None for no momentum.
        Default is [None, 1e-2, 1e-1].
    Returns:
    -------
    list[tuple]
        A list of tuples representing the results for each hyperparameter combination.
        Each tuple contains:
            - Activation function name (str).
            - Learning rate (η).
            - Momentum value (γ).
            - Pool size.
            - A list of validation accuracy percentages over the runs.

    Example:
    --------
    result = search_hyperparameters(
        learning_rates=[0.001, 0.01],
        momentums=[0.9, None],
        pools=[32],
        runs=50,
        epoch_unit=5
    )
    """
    
    n = len(J)
    H = 0.0
    for l in range(0, n):
        si , sj = S[l],  S[l+1]
        W = J[l]
        H -= np.sum((W @ si) * sj)

    return H


def test_hamiltonian():
    
    spins = [np.array([1, -1, 1]),
             np.array([1, -1, 1, 1]),
             np.array([-1, 1])]

    weight_matrices = [np.array([[3, -3, 1],
                                 [5, -2, 1],
                                 [6, 8, 1],
                                 [-1, -1, -1]]),

                       np.array([[-2, 7, 9, 4],
                                 [1, 3, -5, 1]])]
    
    H = calculate_hamiltonian(weight_matrices, spins)
    hand_calculation = 13.0

    assert H == hand_calculation