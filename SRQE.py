import torch
import numpy as np
from Qcontent_function import Q_content_calculate
from Qstyle_function import Q_style_calculate

from torchvision import models,transforms

def prepare_image(image):
    image = transforms.ToTensor()(image)
    #image = torch.tensor(image, dtype = torch.float64)
    return image.unsqueeze(0)


def SRQE(img_content,img_style,img_result,use_gpu):
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Convert color images to grayscale images
    img_c = np.array(img_content.convert('L'), dtype=np.float64)
    img_r = np.array(img_result.convert('L'), dtype=np.float64)
    Q_content = Q_content_calculate(img_c,img_r)
    
    img_style = prepare_image(img_style).to(device)
    img_result = prepare_image(img_result).to(device)
    Q_style = Q_style_calculate(img_style,img_result,use_gpu)
    Q_overall = (Q_content ** 0.4) * (Q_style ** 0.6)
    
    return Q_content,Q_style,Q_overall