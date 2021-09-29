import os
import torch.utils.data as data
import numpy as np
import pickle
import imageio
from pathlib import Path

from .augment import augment, augment_config, augment_landmark


def imread(x): return np.asarray(imageio.imread(x))

# TODO: train with multiple source

class FlowerDataset(data.dataset.Dataset):
    def __init__(self, img1_dir, img2_dir,
                 sf, img1_keys, img2_keys,
                 landmark=None, landmark_reverse=False,
                 names_path=None, augment=True):

        self.img1_dir = Path(img1_dir)
        self.img2_dir = Path(img2_dir)
        self.img1_keys = img1_keys
        self.img2_keys = img2_keys

        self.sf = sf
        self.landmark = landmark
        self.landmark_reverse = landmark_reverse
        self.augment = augment

        # get the name of all images
        if names_path is None:
            self.names = os.listdir(os.path.join(img1_dir, 'HR'))
        else:
            with open(names_path, 'r') as f:
                names = f.readlines()
                self.names = [n.strip() for n in names]

    def __getitem__(self, index):
        name = self.names[index]

        data = {}
        data['img1_HR'] = imread(self.img1_dir / 'HR' / name)
        for k in self.img1_keys:
            data[f'img1_{k}'] = imread(self.img1_dir / f'sf_{self.sf}' / k / name)

        data['img2_HR'] = imread(self.img2_dir / 'HR' / name)
        for k in self.img2_keys:
            data[f'img2_{k}'] = imread(self.img2_dir / f'sf_{self.sf}' / k / name)

        for k, v in data.items():
            data[k] = v.transpose(2, 0, 1).astype('float32') / 255

        if self.augment:
            config = augment_config()
            for k, v in data.items():
                data[k] = augment(v, config)

        if self.landmark:
            with open(os.path.join(self.landmark, name[:-4]+'.pkl'), "rb") as fp:
                landmarks = pickle.load(fp)
            if self.augment:
                _, w, h = data['img1_HR'].shape
                landmarks = augment_landmark(landmarks, w, h, config)
            landmarks = np.array(landmarks)
            if self.landmark_reverse:
                rlandmarks = np.ones_like(landmarks)
                rlandmarks[:,:2] = landmarks[:,2:]
                rlandmarks[:,2:] = landmarks[:,:2]
                landmarks = rlandmarks
            data['landmark'] = landmarks

        return data

    def __len__(self):
        return len(self.names)
