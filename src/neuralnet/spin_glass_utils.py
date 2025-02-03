import numpy as np
from neuralnet.neuralnet import NeuralNetwork
from neuralnet.activations import step


def map_nn_to_spin_glass(x: np.ndarray[:], spin_net: NeuralNetwork) -> list[np.ndarray]:
    """
    Map a neural network to an Ising-model spin-glass with external magnetic field.

    Args:
        x (array): Input data point.
        spin_net (object): Neural network with weights and biases.

    Returns:
        list: Spin configurations at each layer of the network.
    """
    # Initialize spin configurations
    spins = [np.where(step(x) > .5, 1, -1)]  # Input layer spins
    
    # Iterate over the layers of the network
    activation = spins[0]
    for l in range(spin_net.layer_n - 1):
        # Compute activation and determine spins for the current layer
        activation = step(spin_net.weights[l] @ activation + spin_net.bias[l])
        spins.append(np.where(activation > 0, 1, -1))
    
    return spins


def calculate_hamiltonian(J: list[np.ndarray[:,:]], S: list[np.ndarray[:]]) -> (int | float):

    """
    Calculates the hamiltonian for a multi-layer spin-glass model

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


def form_J_matrix(W):
    """
    Constructs a block matrix from a list of lists, where non-zero elements
    are actual matrices and zero elements are placeholders.

    Args:
        block_list (list[list[np.ndarray or int]]): A 2D list where:
            - Non-zero elements are actual matrices.
            - Zero elements represent placeholders for zero matrices.

    Returns:
        np.ndarray: The constructed block matrix.
    """
    # Determine the number of blocks (rows and columns in the structure)
    n = len(W)
    block_list = [
        [0 for i in range(0, n + 1)] for i in range(0, n + 1)
    ]  # an nxn matrix with zeros

    for i in range(n):  # Loop through consecutive pairs
        block_list[i][i + 1] = W[i].T
        block_list[i + 1][i] = W[i]  # Lower diagonal

    # Determine the sizes of each block row and block column
    row_sizes = [max(block.shape[0] if isinstance(block, np.ndarray) else 0 for block in row) for row in block_list]
    col_sizes = [max(block.shape[1] if isinstance(block, np.ndarray) else 0 for block in col) for col in zip(*block_list)]
    
    # Construct the full block matrix
    block_rows = []
    for i, row in enumerate(block_list):
        block_row = []
        for j, block in enumerate(row):
            if isinstance(block, np.ndarray):
                # Non-zero matrix: Use the actual matrix
                block_row.append(block)
            else:
                # Zero placeholder: Create a zero matrix of the appropriate size
                block_row.append(np.zeros((row_sizes[i], col_sizes[j])))
        block_rows.append(block_row)
    
    # Use np.block to combine the block rows into the final block matrix
    J = np.block(block_rows)
    return J 


def form_bond_matrix(β: (float | int), J : np.array) -> np.array:
    """form M matrix to analyze phase transition of spin glass

    Parameters
    ----------
    β  : (float | int)
        inverse temperature
    J : np.array
        bond matrix

    Returns
    -------
    np.array
        The M matrix allowing analysis of phase of phase transition

    """

    # check symmetric 
    assert np.allclose(J.T,J)

    # scale J by beta
    Jβ = J*β
    M = Jβ-β*np.diag(np.sum(Jβ**2, axis=1))
    return M 