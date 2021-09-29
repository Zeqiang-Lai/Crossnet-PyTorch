import os
import argparse
from pathlib import Path
import cv2
from torchlight.utils.transforms import GaussianDownsample, Upsample


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', required=True, type=str, help='data root')
parser.add_argument('-sf', type=int, default=4, help='scale factor')

args = parser.parse_args()

print('root: ' + args.root)
print('Scale Factor: ' + str(args.sf))

sf = args.sf
downsample = GaussianDownsample(sf, ksize=8, sigma=3)
bicubic_upsample = Upsample(sf, mode='cubic')


def process(root):
    lr_dir = root / f'sf_{sf}' / 'LR'
    bi_dir = root / f'sf_{sf}' / 'SR_bicubic'

    lr_dir.mkdir(exist_ok=True, parents=True)
    bi_dir.mkdir(exist_ok=True, parents=True)

    hr_dir = root / 'HR'
    
    names = os.listdir(hr_dir)
    names = filter(lambda x: x.endswith('.png'), names)
    names = sorted(names)

    for idx, name in enumerate(names):
        path = hr_dir / name
        print(f'{idx}|{len(names)}: {path}')

        hr = cv2.imread(str(path))

        lr = downsample(hr)
        bicubic = bicubic_upsample(lr)

        cv2.imwrite(str(lr_dir / name), lr)
        cv2.imwrite(str(bi_dir / name), bicubic)


viewpoints = [(0, 0), (1, 1), (3, 3), (7, 7)]
for x, y in viewpoints:
    root = Path(args.root)
    root = root / f'{x}_{y}'
    print(root)
    process(root)
