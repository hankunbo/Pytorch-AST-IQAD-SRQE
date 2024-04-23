
import numpy as np
import torch
from .compute_SS import compute_SS  # Assuming you have defined compute_SS function
from .gram_data_extract import gram_data_extract  # Assuming you have defined gram_data_extract function
from scipy.io import loadmat

def Q_style_calculate(img_s, img_r, use_gpu):
    """
    Calculate the style quality measure.

    Parameters:
        img_s (numpy.ndarray): Style image.
        img_r (numpy.ndarray): Stylized image.
        use_gpu (bool): Whether to use GPU for processing.

    Returns:
        float: Style quality measure.
    """
    Gram_style = gram_data_extract(img_s, use_gpu)
    Gram_stylized = gram_data_extract(img_r, use_gpu)

    # Loading dictionaries
    SD1 = loadmat('./Qstyle_function/SD/SD1.mat')['SP1_VGG2_64_256']
    SD2 = loadmat('./Qstyle_function/SD/SD2.mat')['D']
    SD3 = loadmat('./Qstyle_function/SD/SD3.mat')['SP1_VGG4_256_512']
    SD4 = loadmat('./Qstyle_function/SD/SD4.mat')['D']
    SD5 = loadmat('./Qstyle_function/SD/SD5.mat')['D']

    # Sparse Feature Similarity Measure
    output1 = compute_SS(Gram_style[0], Gram_stylized[0], SD1)
    output2 = compute_SS(Gram_style[1], Gram_stylized[1], SD2)
    output3 = compute_SS(Gram_style[2], Gram_stylized[2], SD3)
    output4 = compute_SS(Gram_style[3], Gram_stylized[3], SD4)
    output5 = compute_SS(Gram_style[4], Gram_stylized[4], SD5)
    Qstyle = output1 * output2 * output3 * output4 * output5
    return Qstyle

# Example usage
# Assuming img_s and img_r are numpy arrays representing the style image and the stylized image respectively
# Q_style = Q_style_calculate(img_s, img_r, use_gpu)
