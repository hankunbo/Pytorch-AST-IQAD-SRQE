# Pytorch-AST-IQAD-SRQE
Pytorch_version of AST-IQAD-SRQE

## Acknowledgement
Our code is overwritten from AST-IQAD-SRQE[https://github.com/Hangwei-Chen/AST-IQAD-SRQE?tab=readme-ov-file]

Our code is borrowed parts from DISTS[https://github.com/dingkeyan93/DISTS] and PCRL[https://web.xidian.edu.cn/ldli/paper.html]. Thanks to them!

## Requirement
torch, PIL, numpy, scipy, torchvision   

## How to run
```
python demo.py
```

## Existing problems
The data in our code is of type float32, while the data type used in the original author's Matlab version of the code is float64, which might causes the deviation in the Style resemblance score after 6 decimal places. Meanwhile, because the Overall vision score is calculated by the Style resemblance score, there may be deviations after 4 decimal places in the Overall vision score.  
Welcome to everyone to fix defects in our code !!!
