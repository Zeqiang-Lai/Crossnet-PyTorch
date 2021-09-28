import os
from pathlib import Path
import random

root = Path('/media/exthdd/datasets/refsr/LF_Flowers_Dataset/Flowers_8bit/')
target = root.parent / 'processed'

names = os.listdir(root)
names = filter(lambda x: x.endswith('.png'), names)
names = sorted(names)

random.seed(2021)
random.shuffle(names)
test = names[:100]
train = names[100:]

with open(target / 'train.txt', 'w') as f:
    f.write('\n'.join(train))
    
with open(target / 'test.txt', 'w') as f:
    f.write('\n'.join(test))