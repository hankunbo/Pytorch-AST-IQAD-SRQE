import numpy as np
from scipy.io import loadmat
from Qcontent_function.DOG_img_extract import DOG_img_extract  # Assuming you have a separate file for DOG_img_extract function
from Qcontent_function.compute_CS import compute_CS  # Assuming you have a separate file for compute_CS function

def Q_content_calculate(Raw_content, stylized):
    scale = 3

    # DOG decomposition
    ref_DOG_octave1, ref_DOG_octave2, ref_DOG_octave3, ref_DOG_octave4 = DOG_img_extract(Raw_content, scale)
    stylized_DOG_octave1, stylized_DOG_octave2, stylized_DOG_octave3, stylized_DOG_octave4 = DOG_img_extract(stylized, scale)
    
    # Loading dictionary
    DOG_octave1_CD = loadmat('./Qcontent_function/CD/CD1.mat')['DOG_octave1_D']
    DOG_octave2_CD = loadmat('./Qcontent_function/CD/CD2.mat')['DOG_octave2_D']
    DOG_octave3_CD = loadmat('./Qcontent_function/CD/CD3.mat')['DOG_octave3_D']
    DOG_octave4_CD = loadmat('./Qcontent_function/CD/CD4.mat')['DOG_octave4_D']
    
    Q1 = np.zeros(scale)
    for band in range(scale):
        D1 = DOG_octave1_CD[:, :, band]
        Q1[band] = compute_CS(ref_DOG_octave1[band], stylized_DOG_octave1[band], D1)
    Out_Q1 = np.mean(Q1)
    

    Q2 = np.zeros(scale)
    for band in range(scale):
        D2 = DOG_octave2_CD[:, :, band]
        Q2[band] = compute_CS(ref_DOG_octave2[band], stylized_DOG_octave2[band], D2)
    Out_Q2 = np.mean(Q2)

    Q3 = np.zeros(scale)
    for band in range(scale):
        D3 = DOG_octave3_CD[:, :, band]
        Q3[band] = compute_CS(ref_DOG_octave3[band], stylized_DOG_octave3[band], D3)
    Out_Q3 = np.mean(Q3)

    Q4 = np.zeros(scale)
    for band in range(scale):
        D4 = DOG_octave4_CD[:, :, band]
        Q4[band] = compute_CS(ref_DOG_octave4[band], stylized_DOG_octave4[band], D4)
    Out_Q4 = np.mean(Q4)
    
    output = Out_Q1 * Out_Q2 * Out_Q3 * Out_Q4

    return output


