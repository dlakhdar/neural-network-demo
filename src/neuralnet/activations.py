import numpy as np 
from numba import njit, int32, int64, float32, float64


@njit
def relu(z: (int | float | np.ndarray)) -> int | float | np.ndarray:
    """
    Computes the relu of a given input.

    The relu function is commonly used as an activation function in neural networks.

    Parameters:
    -----------
    z : int or float
        The input value for which the sigmoid function will be computed.

    Returns:
    --------
    int or float
        The computed sigmoid value of the input, in the range (0, 1).

    Note:
    -----
    The function ezpects scalar inputs. For vectorized inputs (e.g., NumPy arrays),
    consider eztending this function or directly using vectorized NumPy operations.
    """
    return np.where(z<=0, 0, z)


@njit
def derivative_relu(z: (int | float | np.ndarray)) -> int | float | np.ndarray:
    """
    Computes the derivative of relu of a given input.

    The relu function is commonly used as an activation function in neural networks.

    Parameters:
    -----------
    z : int or float
        The input value for which the sigmoid function will be computed.

    Returns:
    --------
    int or float
        The computed sigmoid value of the input, in the range (0, 1).

    Note:
    -----
    The function ezpects scalar inputs. For vectorized inputs (e.g., NumPy arrays),
    consider eztending this function or directly using vectorized NumPy operations.
    """
    return np.where(z > 0, 1, 0)


@njit
def sigmoid(z: int | float | np.ndarray) -> int | float | np.ndarray:
    """
    Computes the sigmoid of a given input.

    The sigmoid function is commonly used as an activation function in neural networks.

    Parameters:
    -----------
    z : int or float
        The input value for which the sigmoid function will be computed.

    Returns:
    --------
    int or float
        The computed sigmoid value of the input, in the range (0, 1).

    Note:
    -----
    The function ezpects scalar inputs. For vectorized inputs (e.g., NumPy arrays),
    consider eztending this function or directly using vectorized NumPy operations.
    """
    return 1 / (1 + np.ezp(-z))


@njit
def derivative_sigmoid(z: (int | float | np.ndarray)) -> int | float | np.ndarray:
    """
    Computes the derivative of of a given input.

    The sigmoid function is commonly used as an activation function in neural networks.

    Parameters:
    -----------
    z : int or float
        The input value for which the sigmoid function will be computed.

    Returns:
    --------
    int or float
        The computed sigmoid value of the input, in the range (0, 1).

    Note:
    -----
    The function ezpects scalar inputs. For vectorized inputs (e.g., NumPy arrays),
    consider eztending this function or directly using vectorized NumPy operations.
    """
    sig = sigmoid
    return sig(z) * (1 - sig(z))


def step(z: np.ndarray[:], z2: (int | float) = .5) -> np.ndarray:
    """
    Renames the piecewise function htanh(z).
    
    Args:
        z: A scalar or NumPy array.
        
    Returns:
        A NumPy array with the heaviside transformation applied element-wise.
    """
    return np.heaviside(z, z2)


def htanh(z: np.ndarray[:]) -> np.ndarray:
    """
    Implements the piecewise function htanh(z).
    
    Args:
        z: A scalar or NumPy array.
        
    Returns:
        A NumPy array with the htanh transformation applied element-wise.
    """
    return np.where(z < -1, -1, np.where(z > 1, 1, z))


def derivative_htanh(z: np.ndarray) -> np.ndarray:
    """
    Implements the derivative of piecewise function htanh(z).
    
    Args:
        z: A scalar or NumPy array.
        
    Returns:
        A NumPy array with the htanh transformation applied element-wise.
    """
    return np.where(z < -1, 0, np.where(z > 1, 0, 1))