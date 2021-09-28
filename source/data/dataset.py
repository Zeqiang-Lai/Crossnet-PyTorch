import os
import cv2
import torch.utils.data as data
from scipy.ndimage.interpolation import zoom
import numpy as np
import pickle

# https://github.com/pratulsrinivasan/Local_Light_Field_Synthesis


def grayscale(img):
    dst = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)
    dst[0, :, :] = 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]
    dst = np.repeat(dst, 3, axis=0)
    return dst


def blend(img1, img2, alpha=0.5):
    return img1 * alpha + img2 * (1-alpha)


class FlowerDataset(data.dataset.Dataset):
    def __init__(self, path, return_name=False, augment=True):
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
        img1_HR = cv2.imread(self.img1_HR_paths[index]).transpose(2, 0, 1).astype('float32') / 255
        img1_LR = cv2.imread(self.img1_LR_paths[index]).transpose(2, 0, 1).astype('float32') / 255
        img1_SR = cv2.imread(self.img1_SR_paths[index]).transpose(2, 0, 1).astype('float32') / 255
        img2_HR = cv2.imread(self.img2_HR_paths[index]).transpose(2, 0, 1).astype('float32') / 255
        img2_LR = cv2.imread(self.img2_LR_paths[index]).transpose(2, 0, 1).astype('float32') / 255

        with open(self.matches_paths[index], "rb") as fp:
            matches = pickle.load(fp)

        if self.augment:
            config = self._augment_config()
            img1_HR = self._augment(img1_HR, config)
            img1_LR = self._augment(img1_LR, config)
            img1_SR = self._augment(img1_SR, config)
            img2_HR = self._augment(img2_HR, config)
            img2_LR = self._augment(img2_LR, config)
            matches = self._augment_match(img1_HR, matches, config)
        matches = np.array(matches)
        
        if self.return_name:
            name = os.path.basename(self.img1_HR_paths[index])
            return (img1_LR, img1_HR, img1_SR, img2_HR, img2_LR), matches, name
        else:
            return (img1_LR, img1_HR, img1_SR, img2_HR, img2_LR), matches

    def _augment_config(self):
        config_flip = 0
        config_flip_lr = np.random.randint(0, 2) > 0.5
        config_brightness_changes = np.random.normal(loc=0, scale=0.02)
        config_multiplicative_color_changes = np.random.uniform(0.9, 1.1)
        config_contrast = np.random.uniform(-0.3, 0.3)
        config_gamma = np.random.uniform(0.8, 1.3)
        config = [config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes, config_contrast, config_gamma]
        return config

    def _augment_match(self, img, matches, config):
        config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes, config_contrast, config_gamma = config
        c, w, h = img.shape

        new = []
        for match in matches:
            x1, y1, x2, y2 = match
            if config_flip:
                y1 = h - y1
                y2 = h - y2

            if config_flip_lr:
                x1 = w - x1
                x2 = w - x2
            new.append((x1, y2, x2, y2))
        return new

    def _augment(self, img, config):
        config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes, config_contrast, config_gamma = config

        img_aug = img
        if(config_flip == 1):
            img_aug = img_aug[..., ::-1, :]
        if(config_flip_lr == 1):
            img_aug = img_aug[..., :, ::-1]

        # brightness changes
        img_aug = img_aug + config_brightness_changes
        # multiplicative color changes
        img_aug = img_aug * config_multiplicative_color_changes

        # ## Contrast
        gs_2 = grayscale(img_aug)
        img_aug = blend(gs_2, img_aug, alpha=config_contrast)
        # clip
        img_aug = np.clip(img_aug, 0.0, 1.0)

        return img_aug

    def __len__(self):
        return len(self.img1_HR_paths)


class FlowerDataset3(data.dataset.Dataset):
    def __init__(self, path):
        names = os.listdir(os.path.join(path, '4_3'))
        self.img1_paths = [os.path.join(path, '4_3', name) for name in names]
        self.img2_paths = [os.path.join(path, '3_3', name) for name in names]
        self.sf = 4

    def __getitem__(self, index):
        img1 = cv2.imread(self.img1_paths[index])[28:28+320, 14:526, :].transpose(2, 0, 1).astype('float32')
        img2 = cv2.imread(self.img2_paths[index])[28:28+320, 14:526, :].transpose(2, 0, 1).astype('float32')
        img1 /= 255
        img2 /= 255

        img1_LR = self._downsample(img1)
        img1_HR = img1
        img1_SR = img1_LR

        img2_LR = self._downsample(img2)
        img2_HR = img2
        return img1_LR, img1_HR, img1_SR, img2_HR, img2_LR

    def _downsample(self, x):
        return zoom(zoom(x, (1, 1/self.sf, 1/self.sf)), (1, self.sf, self.sf))

    def __len__(self):
        return len(self.img1_paths)


class FlowerDataset2(data.dataset.Dataset):
    def __init__(self, path):
        self.paths = [os.path.join(path, name) for name in os.listdir(path)]
        self.sf = 4

    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(path)

        imgs = []
        for i in range(3, 11):
            for j in range(3, 11):
                view = img[i:, j:, :][::14, ::14, :]  # 376, 540, 3
                view = view[28:28+320, 14:526, :]
                imgs.append(view)

        img1, img2 = imgs[0], imgs[1]

        img1_LR = self._downsample(img1)
        img1_HR = img1
        img1_SR = img1_LR

        img2_LR = self._downsample(img2)
        img2_HR = img2
        return img1_LR, img1_HR, img1_SR, img2_HR, img2_LR

    def _downsample(self, x):
        return zoom(zoom(x, (1/self.sf, 1/self.sf, 1)), (self.sf, self.sf, 1))

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    path = '/media/exthdd/datasets/LF_Flowers_Dataset/'
    dataset = FlowerDataset(path)
    img1_LR, img1_HR, img1_SR, img2_HR, img2_LR = dataset.__getitem__(10)
    print(img1_LR.shape)
    print(img1_HR.shape)
    print(img1_SR.shape)
    print(img2_HR.shape)
    print(img2_LR.shape)
