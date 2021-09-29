# CrossNet PyTorch

This repository reimplements the training pipeline of CrossNet and provides an unoffical implemenation of CrossNet++.

The offical implemenation of CrossNet: [ECCV2018_CrossNet_RefSR](https://github.com/htzheng/ECCV2018_CrossNet_RefSR)

Reference Papers:

1. CrossNet: An End-to-end Reference-based Super Resolution Network using Cross-scale Warping
2. CrossNet++: Cross-scale Large-parallax Warping for Reference-based Super-resolution

## Requirements

- PyTorch
- Python3 (tested on Python3.7)
- torchlight

## Getting Started

### Datasets

Download the original light field datasets, [Flower](https://github.com/pratulsrinivasan/Local_Light_Field_Synthesis), [LFVideo](https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/SIG17/lfv/).

### Train

```shell
python run.py train -s saved/flower/crossnet/base -c configs/flower/crossnet.yaml
python run.py train -s saved/flower/crossnet++/base -c configs/flower/crossnet++.yaml
```

### Test

```shell
python run.py test -s saved/flower/crossnet/base -r best
python run.py test -s saved/flower/crossnet++/base -r best
```

### Results

- Serpate model for each viewpoint. Reference image are at (0,0).
- Charbonnier loss only

| Model      | Scale | ViewPoint | PSNR  |
| ---------- | ----- | --------- | ----- |
| CrossNet   | 4     | 1,1       | 42.05 |
|            |       | 3,3       |       |
|            |       | 7,7       |       |
| CrossNet++ | 4     | 1,1       |       |
|            |       | 3,3       |       |
|            |       | 7,7       |       |

## Reference

- [EDSR]()