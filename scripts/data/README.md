# Data Process

1. decompose different viewpoints of `Flowers_8bit` into seperate image.

2. landmark

```shell
python landmark.py -img1 /home/wzliu/lzq/data/flower/0_0/HR -img2 /home/wzliu/lzq/data/flower/3_3/HR -o /home/wzliu/lzq/data/flower/landmarks/0_0_3_3
```

The format of the dataset:

```txt
flower
  - 0_0
    - HR
    - LR
    - LR_bicubic
    - SR_[model]
  - 1_1
    - HR
    - LR
    - LR_bicubic
    - SR_[model]
```
