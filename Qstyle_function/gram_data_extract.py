
import numpy as np
import torch
from torchvision import transforms
from .DISTS_pt import DISTS



def gram_data_extract(img, use_gpu):
    """
    Extract gram matrices from the input image.

    Parameters:
        img (numpy.ndarray): Input image.
        use_gpu (bool): Whether to use GPU for processing.

    Returns:
        list: List of gram matrices.
    """
    Gram_cell = []
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model = DISTS(load_weights = True).to(device)
    with torch.no_grad():
        stylized_features = model.forward_once(img)
    #print(stylized_features[0])
    # Assume extractdata function extracts the data from stylized_features
    # fd1 = extractdata(stylized_features[0])
    fd2 = stylized_features[1]
    fd3 = stylized_features[2]
    fd4 = stylized_features[3]
    fd5 = stylized_features[4]
    fd6 = stylized_features[5]
    for fd in [fd2, fd3, fd4, fd5, fd6]:
        fd = fd.squeeze(0).permute(2, 1, 0).cpu().numpy()
        A = np.zeros((fd.shape[2], fd.shape[0] * fd.shape[1]))
        for i in range(fd.shape[2]):
            R = fd[:, :, i]
            Ii = R.reshape(-1)
            A[i, :] = Ii
        Gram = np.dot(A, A.T)
        Gram_cell.append(Gram)

    return Gram_cell
