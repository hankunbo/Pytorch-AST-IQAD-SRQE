

import numpy as np
from numpy.linalg import pinv

def compute_SS(style_gram, stylized_gram, D):
    """
    Compute the style similarity measure.

    Parameters:
        style_gram (numpy.ndarray): Style gram matrix.
        stylized_gram (numpy.ndarray): Stylized gram matrix.
        D (numpy.ndarray): Dictionary.

    Returns:
        float: Style similarity measure.
    """
    c = 0.02
    #pinv伪逆
    rcoef = pinv(D) @ style_gram
    dcoef = pinv(D) @ stylized_gram

    SS = np.mean((np.sum(np.abs(rcoef * dcoef), axis=0) + c) / ((np.sum(rcoef**2, axis=0) * np.sum(dcoef**2, axis=0))**0.5 + c))

    return SS

# Example usage
# Assuming style_gram, stylized_gram, and D are numpy arrays
# SS_value = compute_SS(style_gram, stylized_gram, D)
