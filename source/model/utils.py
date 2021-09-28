import torch.nn as nn


def init_weights(w, init_type):
    if init_type == 'w_init_relu':
        nn.init.kaiming_uniform_(w, nonlinearity='relu')
    elif init_type == 'w_init_leaky':
        nn.init.kaiming_uniform_(w, nonlinearity='leaky_relu')
    elif init_type == 'w_init':
        nn.init.uniform_(w)


def Activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif activation == 'selu':
        return nn.SELU(inplace=True)
    elif activation == 'linear':
        return nn.Identity()


def conv_activation(in_ch, out_ch, kernel_size=3, stride=1, padding=1, activation='relu', init_type='w_init_relu'):
    block = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        Activation(activation)
    )
    return block


def flow(in_ch, out_ch, kernel_size=3, stride=1, padding=1, activation='linear', init_type='w_init'):
    block = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        Activation(activation)
    )
    return block


def upsample(in_ch, out_ch):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True)


def leaky_deconv(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv_activation(in_ch, out_ch, activation='relu'):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
        Activation(activation)
    )
    return block
