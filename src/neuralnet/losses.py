import numpy as np 
from numba import njit


@njit
def mse(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (a - y)**2/2


@njit
def mse_grad(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return a - y


def l2_hinge_loss_gradient(activations, one_hot_label):
    """
    Computes the gradient of L2 multi-class hinge loss for a single instance.

    Args:
        activations (np.ndarray): Activations (logits) for a single instance (shape: (C,)).
        one_hot_label (np.ndarray): One-hot encoded label (shape: (C,)).

    Returns:
        np.ndarray: Gradient of the L2 hinge loss with respect to activations (shape: (C,)).
    """
    # Find the index of the correct class
    y_index = np.argmax(one_hot_label)
    
    # Initialize gradient
    grad = np.zeros_like(activations)
    
    # Compute margin for each class
    for j in range(len(activations)):
        if j == y_index:
            continue  # Skip the correct class for now
        
        margin = 1 + activations[j] - activations[y_index]
        if margin > 0:
            grad[j] = 2 * margin
            grad[y_index] -= 2 * margin  # Accumulate gradient for the correct class
    
    return grad
