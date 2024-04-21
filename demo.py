from PIL import Image
from SRQE import SRQE

img_content = Image.open('content.jpg')
img_style = Image.open('style.jpg')
img_result = Image.open('stylized_AdaIN.png')


use_gpu = 1 # 0:No, 1:Yes; Using GPU can save a lot of time

Q_content,Q_style,Q_overall = SRQE(img_content,img_style,img_result,use_gpu)

print('Content preservation score = {:.4f}'.format(Q_content))
print('Style resemblance score = {:.4f}'.format(Q_style))
print('Overall vision score = {:.4f}'.format(Q_overall))
