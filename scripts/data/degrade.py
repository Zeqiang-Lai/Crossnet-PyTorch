import os
from pathlib import Path
import cv2
from torchlight.utils.transforms import GaussianDownsample, Upsample

sf = 4
downsample = GaussianDownsample(sf, ksize=8, sigma=3)
bicubic_upsample = Upsample(sf, mode='cubic')


def process(root):
    lr_target = root.parent / f'sf_{sf}' / 'LR'
    bi_target = root.parent / f'sf_{sf}' / 'Bicubic'

    lr_target.mkdir(parents=True)
    bi_target.mkdir(parents=True)

    names = os.listdir(root)
    names = filter(lambda x: x.endswith('.png'), names)
    names = sorted(names)

    for idx, name in enumerate(names):
        print(f'{idx}|{len(names)}: {name}')

        path = root / name
        hr = cv2.imread(str(path))

        lr = downsample(hr)
        bicubic = bicubic_upsample(lr)

        cv2.imwrite(str(lr_target / name), lr)
        cv2.imwrite(str(bi_target / name), bicubic)


viewpoints = [(0, 0), (1, 1), (3, 3), (7, 7)]
for x, y in viewpoints:
    root = Path('/media/exthdd/datasets/refsr/LF_Flowers_Dataset/processed')
    root = root / f'{x}_{y}'
    process(root)
