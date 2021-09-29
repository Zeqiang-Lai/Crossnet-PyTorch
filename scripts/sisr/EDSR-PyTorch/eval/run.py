# %%
import os
from pathlib import Path

import imageio
import torch
from edsr import EDSR

sf = 4
# baseline model
model = EDSR(scale=sf).cuda()
# paper model
# model = EDSR(scale=4, n_resblocks=32, n_feats=256, res_scale=0.1).cuda()

path = '../experiment/flower/edsr_baseline_x4/model/model_best.pt'
# path = '../experiment/flower/edsr_x4/model/model_best.pt'
model.load_state_dict(torch.load(path))
model.eval()


def upsample(x):
    x = torch.from_numpy(x).float()
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0).cuda()
    o = model(x)
    o = o.squeeze()
    o = o.permute(1, 2, 0)
    o = o.detach().cpu().numpy()
    o = o.clip(0,255).astype('uint8')
    return o


root = Path('/data/home/wzliu/lzq/data/flower/0_0')
lr_dir = root / f'sf_{sf}' / 'LR'
hr_dir = root / 'HR'
sr_dir = root / f'sf_{sf}' / 'SR_EDSR_baseline'
sr_dir.mkdir(exist_ok=True)

names = os.listdir(lr_dir)
names = filter(lambda x: x.endswith('.png'), names)
names = sorted(names)

for idx, name in enumerate(names):
    print(idx, len(names), name)
    lr = imageio.imread(lr_dir / name)
    hr = imageio.imread(lr_dir / name)
    sr = upsample(lr)
    imageio.imwrite(sr_dir / name, sr)
    
    # import matplotlib.pyplot as plt
    # plt.imshow(sr)
    # break

# %%
