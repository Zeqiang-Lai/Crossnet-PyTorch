import numpy as np


def grayscale(img):
    dst = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)
    dst[0, :, :] = 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]
    dst = np.repeat(dst, 3, axis=0)
    return dst


def blend(img1, img2, alpha=0.5):
    return img1 * alpha + img2 * (1-alpha)


def augment_config():
    config_flip = 0
    config_flip_lr = np.random.randint(0, 2) > 0.5
    config_brightness_changes = np.random.normal(loc=0, scale=0.02)
    config_multiplicative_color_changes = np.random.uniform(0.9, 1.1)
    config_contrast = np.random.uniform(-0.3, 0.3)
    config_gamma = np.random.uniform(0.8, 1.3)
    config = [config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes, config_contrast, config_gamma]
    return config


def augment(img, config):
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


def augment_landmark(landmarks, w, h, config):
    config_flip, config_flip_lr, config_brightness_changes, config_multiplicative_color_changes, config_contrast, config_gamma = config
    new = []
    for landmark in landmarks:
        x1, y1, x2, y2 = landmark
        if config_flip:
            y1 = h - y1
            y2 = h - y2

        if config_flip_lr:
            x1 = w - x1
            x2 = w - x2
        new.append((x1, y2, x2, y2))
    return new
