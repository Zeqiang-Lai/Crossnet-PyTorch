import os
import torch.utils.data as data
import numpy as np
import pickle
import imageio
from pathlib import Path

from .augment import augment, augment_config, augment_landmark


def imread(x): return np.asarray(imageio.imread(x))


class FlowerDataset2(data.dataset.Dataset):
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


class FlowerDataset(data.dataset.Dataset):
    def __init__(self, path, mode, return_name=False, augment=True):
        names = os.listdir(os.path.join(path, 'img1_HR'))
        self.img1_HR_paths = [os.path.join(path, 'img1_HR', name) for name in names]
        self.img1_LR_paths = [os.path.join(path, 'img1_LR', name) for name in names]
        self.img1_SR_paths = [os.path.join(path, 'img1_SR', name) for name in names]
        self.img2_HR_paths = [os.path.join(path, 'img2_HR', name) for name in names]
        self.img2_LR_paths = [os.path.join(path, 'img2_LR', name) for name in names]
        self.matches_paths = [os.path.join(path, 'img1_img2_matches', name[:-4]+'.pkl') for name in names]
        self.return_name = return_name
        self.augment = augment

    def __getitem__(self, index):
        img1_HR = imread(self.img1_HR_paths[index]).transpose(2, 0, 1).astype('float32') / 255
        img1_LR = imread(self.img1_LR_paths[index]).transpose(2, 0, 1).astype('float32') / 255
        img1_SR = imread(self.img1_SR_paths[index]).transpose(2, 0, 1).astype('float32') / 255
        img2_HR = imread(self.img2_HR_paths[index]).transpose(2, 0, 1).astype('float32') / 255
        img2_LR = imread(self.img2_LR_paths[index]).transpose(2, 0, 1).astype('float32') / 255

        with open(self.matches_paths[index], "rb") as fp:
            matches = pickle.load(fp)

        if self.augment:
            config = augment_config()
            img1_HR = augment(img1_HR, config)
            img1_LR = augment(img1_LR, config)
            img1_SR = augment(img1_SR, config)
            img2_HR = augment(img2_HR, config)
            img2_LR = augment(img2_LR, config)
            matches = augment_landmark(img1_HR, matches, config)
        matches = np.array(matches)

        if self.return_name:
            name = os.path.basename(self.img1_HR_paths[index])
            return (img1_LR, img1_HR, img1_SR, img2_HR, img2_LR), matches, name
        else:
            return (img1_LR, img1_HR, img1_SR, img2_HR, img2_LR), matches

    def __len__(self):
        return len(self.img1_HR_paths)

    