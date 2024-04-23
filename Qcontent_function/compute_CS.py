
import numpy as np
from numpy.linalg import pinv

def compute_CS(r, d, D):
    """
    Compute the sparse feature similarity measure.

    Parameters:
        r (numpy.ndarray): Content DoG image.
        d (numpy.ndarray): Stylized DoG image.
        D (numpy.ndarray): Dictionary.

    Returns:
        float: Sparse feature similarity measure.
    """
    patchsize = int(np.sqrt(D.shape[0]))
    c = 0.0004

    r = im2col(r, (patchsize, patchsize))
    rcoef = pinv(D) @ r
    
    vt = 0 * np.mean(rcoef**2)
    selectp = np.where(np.mean(rcoef**2, axis=0) > vt)[0]
    rcoef = rcoef[:, selectp]

    d = im2col(d, (patchsize, patchsize))
    dcoef = pinv(D) @ d
    dcoef = dcoef[:, selectp]

    CS = np.mean((np.sum(np.abs(rcoef * dcoef), axis=0) + c) / ((np.sum(rcoef**2, axis=0) * np.sum(dcoef**2, axis=0))**0.5 + c))
    return CS

def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    if mtx_shape[0] % block_size[0] != 0:
        sx = mtx_shape[0] // block_size[0] + 1
    else:
        sx = mtx_shape[0] // block_size[0]

    if mtx_shape[1] % block_size[1] != 0:
        sy = mtx_shape[1] // block_size[1] + 1
    else:
        sy = mtx_shape[1] // block_size[1]

    result = np.empty((block_size[0] * block_size[1], sx * sy))
    
    for i in range(sy):
        for j in range(sx):

            Zeros = np.zeros((block_size[0], block_size[1]))
            block = mtx[j * block_size[0]: j * block_size[0] + block_size[0], i * block_size[1]: i * block_size[1] + block_size[1]]

            if block.shape[0] < Zeros.shape[0] and block.shape[1] == Zeros.shape[1]:
                concat = np.zeros((Zeros.shape[0] - block.shape[0], Zeros.shape[1]))
                block = np.vstack((block, concat))

            if block.shape[1] < Zeros.shape[1] and block.shape[0] == Zeros.shape[0]:
                concat = np.zeros((Zeros.shape[0], Zeros.shape[1] - block.shape[1]))
                block = np.hstack((block, concat))

            if block.shape[0] < Zeros.shape[0] and block.shape[1] < Zeros.shape[1]:
                concat = np.zeros((block_size[0], block_size[1]))
                concat[:block.shape[0], :block.shape[1]] = block
                block = concat

            for line in range(block.shape[0]):
                Zeros[line, :] = block[line, :]
            result[:, i * sx + j] = Zeros.ravel(order='F')
    return result