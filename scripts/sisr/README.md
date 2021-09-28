# Training SISR model

## EDSR

```shell
python main.py --model EDSR --scale 4 --save flower/edsr_baseline_x4 --reset --data_train Flower --data_test Flower --data_range 1-3243/3243-3343 --dir_data /home/wzliu/lzq/data/flower/0_0 --ext img

python main.py --model EDSR --scale 4 --save flower/edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --data_train Flower --data_test Flower --data_range 1-3243/3243-3343 --dir_data /home/wzliu/lzq/data/flower/0_0 --ext img
```
