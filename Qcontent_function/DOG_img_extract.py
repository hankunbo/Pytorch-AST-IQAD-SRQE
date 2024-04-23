import numpy as np
from scipy.ndimage import convolve

def DOG_img_extract(A1, scale):
    S = scale + 1
    sigma = 1.6
    hsize = (20, 20)
    k = 2 ** (1 / S)
    
    # Produce several octaves of gaussian filter
    gauKernel1 = []
    gauKernel2 = []
    gauKernel3 = []
    gauKernel4 = []

    for i in range(S):
        gauKernel1.append(np.array(fspecial_gaussian(hsize, k**(i)*sigma)))   # Octave 1
        gauKernel2.append(np.array(fspecial_gaussian(hsize, 2*k**(i)*sigma))) # Octave 2
        gauKernel3.append(np.array(fspecial_gaussian(hsize, 4*k**(i)*sigma))) # Octave 3
        gauKernel4.append(np.array(fspecial_gaussian(hsize, 8*k**(i)*sigma))) # Octave 4
    
    # Down-sampling the image and set as the next octave of reference image
    A2 = convolve(A1, gauKernel1[S-1], mode='constant')  # s-2
    A2 = A2[::2, ::2]
    A3 = convolve(A2, gauKernel2[S-1], mode='constant')
    A3 = A3[::2, ::2]
    A4 = convolve(A3, gauKernel3[S-1], mode='constant')
    A4 = A4[::2, ::2]
    
    # DoG scale space
    DoGA1 = []
    DoGA2 = []
    DoGA3 = []
    DoGA4 = []

    octaveA1 = []
    octaveA2 = []
    octaveA3 = []
    octaveA4 = []

    for i in range(S-1):
        DoGA1.append(gauKernel1[i+1] - gauKernel1[i])
        DoGA2.append(gauKernel2[i+1] - gauKernel2[i])
        DoGA3.append(gauKernel3[i+1] - gauKernel3[i])
        DoGA4.append(gauKernel4[i+1] - gauKernel4[i])

        octaveA1.append(convolve(A1, DoGA1[i], mode='constant'))
        octaveA2.append(convolve(A2, DoGA2[i], mode='constant'))
        octaveA3.append(convolve(A3, DoGA3[i], mode='constant'))
        octaveA4.append(convolve(A4, DoGA4[i], mode='constant'))

    return octaveA1, octaveA2, octaveA3, octaveA4

def fspecial_gaussian(hsize, sigma):
    h, w = hsize
    y, x = np.mgrid[-(h-1)/2:(h-1)/2 + 1, -(w-1)/2:(w-1)/2 + 1]
    g = np.exp(- (x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

