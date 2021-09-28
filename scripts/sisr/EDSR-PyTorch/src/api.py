import torch
import cv2
import imageio
import numpy as np
from model.edsr import EDSR
from option import args
import data.common as common
import utility

def test_rgb():
    model = EDSR(args).cuda()
    # path = '/home/laizeqiang/Desktop/lzq/projects/ref-sr/sisr/EDSR-PyTorch/experiment/edsr_flower_x4/model/model_best.pt'
    path = '/home/laizeqiang/Desktop/lzq/projects/ref-sr/sisr/EDSR-PyTorch/models/edsr_baseline_x2-1bc95232.pt'
    model.load_state_dict(torch.load(path))
    model.eval()
    
    path = '/home/laizeqiang/Desktop/lzq/projects/ref-sr/sisr/EDSR-PyTorch/test/0853x4.png'
    # path = '/media/exthdd/datasets/LF_Flowers_Dataset/Flower/LR_bicubic/X4/IMG_6513_eslfx4.png'
    # input_numpy = imageio.imread(path)
    # imageio.imwrite('input.png', input_numpy)
    
    # input_numpy = input_numpy.transpose(2,0,1).astype('float32')
    # input = torch.from_numpy(input_numpy).cuda().unsqueeze(0)
    
    lr = imageio.imread(path)
    lr, = common.set_channel(lr, n_channels=3)
    input, = common.np2Tensor(lr, rgb_range=256)
    input = input.unsqueeze(0).cuda()
    print(input.shape)
    # print(input)
    output = model(input)
    sr = utility.quantize(output, 256)
    normalized = sr[0]
    tensor_cpu = normalized.byte().permute(1, 2, 0).cpu().numpy()
    tensor_cpu = tensor_cpu.astype('uint8')
    imageio.imwrite('output.png', tensor_cpu)
   
def test_flower():
    model = EDSR(args).cuda()
    path = '/home/laizeqiang/Desktop/lzq/projects/ref-sr/sisr/EDSR-PyTorch/experiment/edsr_flower_x4/model/model_best.pt'
    model.load_state_dict(torch.load(path))
    model.eval()
    
    # path = '/home/laizeqiang/Desktop/lzq/projects/ref-sr/sisr/EDSR-PyTorch/test/0853x4.png'
    path = '/media/exthdd/datasets/LF_Flowers_Dataset/Flower/LR_bicubic/X4/IMG_6513_eslfx4.png'
    input_numpy = imageio.imread(path)
    imageio.imwrite('input.png', input_numpy)
    
    # input_numpy = input_numpy.transpose(2,0,1).astype('float32')
    # input = torch.from_numpy(input_numpy).cuda().unsqueeze(0)
    
    lr = imageio.imread(path)
    # lr, = common.set_channel(lr, n_channels=3)
    np_transpose = lr.transpose((2, 0, 1))
    # print(np_transpose.shape)
    input = torch.from_numpy(np_transpose).float()
    input = input.unsqueeze(0).cuda()
    print(input.shape)
    # print(input)
    output = model(input)
    normalized = output[0]
    tensor_cpu = normalized.permute(1, 2, 0).detach().cpu().numpy()
    tensor_cpu = tensor_cpu.astype('uint8')
    imageio.imwrite('output.png', tensor_cpu) 

if __name__ == '__main__':
    test_flower()