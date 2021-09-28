import os
from pathlib import Path
import cv2

"""
The Flower dataset [47] contains 3343 ﬂowers and plants lightﬁeld images captured by Lytro ILLUM camera,
whereas each light ﬁeld image has 376 × 541 spatial samples, and 14 × 14 angular samples.
Following [47], we extract the central 8 × 8 grid of angular sample to avoid invalid images,
and randomly divide the dataset into 3243 images for training and 100 images for testing.

Training: the LR and reference images are randomly selected from the 8 × 8 angular grid
Test: the LR images at angular position (i, i), 0 < i ≤ 7 and reference images at position (0, 0) are selected
"""

# ------------------------------------------------------------------------- # 

root = Path('/media/exthdd/datasets/refsr/LF_Flowers_Dataset/Flowers_8bit/')
target_dir = root.parent / 'processed'

names = os.listdir(root)
names = filter(lambda x: x.endswith('.png'), names)
names = sorted(names)

# total: x[0,13], y[0,13]
# center 8x8: x[3,10], y[3,10]

# viewpoints = [(x,y) for x in range(8) for y in range(8)]
viewpoints = [(0, 0), (1, 1), (3, 3), (7, 7)]

for idx, name in enumerate(names):
    print(f'{idx}|{len(names)}: {name}')
    
    path = root / name
    img = cv2.imread(str(path))

    for (x, y) in viewpoints:
        decomposed = img[x+3:, y+3:, :][::14, ::14, :]
        if decomposed.shape != (376, 541, 3):
            print(f'warning: {x},{y}, {decomposed.shape} != (376, 541, 3)')
        # crop
        decomposed = decomposed[:320,:512, :]
        save_path = target_dir / f'{x}_{y}' / 'HR' / name
        save_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(save_path), decomposed)
