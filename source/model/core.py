import torch.nn as nn
from .block import FlowNet, Encoder, Decoder
from .warp import BackwardWarp


class MultiscaleWarpingNet(nn.Module):
    def __init__(self):
        super(MultiscaleWarpingNet, self).__init__()
        self.flownet = FlowNet(6)
        self.warp = BackwardWarp()
        self.encoder = Encoder(3)
        self.decoder = Decoder()

    def forward(self, buff, mode='input_img1_SR'):
        input_img1_LR = buff['input_img1_LR']
        input_img1_SR = buff['input_img1_SR']
        input_img2_HR = buff['input_img2_HR']
        
        if mode == 'input_img2_LR':
            input_img2_LR = buff['input_img2_LR']
            flow = self.flownet(input_img1_LR, input_img2_LR)
        elif mode == 'input_img1_LR':
            flow = self.flownet(input_img1_LR, input_img2_HR)
        elif mode == 'input_img1_SR':
            flow = self.flownet(input_img1_SR, input_img2_HR)
        
        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.encoder(input_img2_HR)

        warp_21_conv1 = self.warp(HR2_conv1, flow_12_1)
        warp_21_conv2 = self.warp(HR2_conv2, flow_12_2)
        warp_21_conv3 = self.warp(HR2_conv3, flow_12_3)
        warp_21_conv4 = self.warp(HR2_conv4, flow_12_4)

        sythsis_output = self.decoder(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1, warp_21_conv2, warp_21_conv3, warp_21_conv4)

        return sythsis_output


class EnhancedMultiscaleWarpingNet(nn.Module):
    def __init__(self):
        super(EnhancedMultiscaleWarpingNet, self).__init__()
        self.flownet1 = FlowNet(6)
        self.flownet2 = FlowNet(6)
        self.warp = BackwardWarp()
        self.encoder = Encoder(3)
        self.decoder = Decoder()

    def forward(self, buff, mode='input_img1_SR'):
        input_img1_LR = buff['input_img1_LR']
        input_img1_SR = buff['input_img1_SR']
        input_img1_HR = buff['input_img1_HR']
        input_img2_HR = buff['input_img2_HR']
        input_img2_LR = buff['input_img2_LR']
        
        if mode == 'input_img2_LR':
            input, ref = input_img1_LR, input_img2_LR
        elif mode == 'input_img1_LR':
            input, ref = input_img1_LR, input_img2_HR
        elif mode == 'input_img1_SR':
            input, ref = input_img1_SR, input_img2_HR
        elif mode == 'input_img1_HR':
            input, ref = input_img1_HR, input_img2_HR
        else:
            raise ValueError('invalid mode: ' + mode)

        warp_ref, flow = self.stage1_align(input, ref)
        warp_21_conv1, warp_21_conv2, warp_21_conv3, warp_21_conv4 = self.stage2_align(input, warp_ref)

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.encoder(input_img1_SR)
        sythsis_output = self.decoder(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_21_conv1, warp_21_conv2, warp_21_conv3, warp_21_conv4)

        return sythsis_output, (warp_ref, [warp_21_conv1, warp_21_conv2, warp_21_conv3, warp_21_conv4]), flow

    def stage1_align(self, x, ref):
        flow = self.flownet1(x, ref)
        flow_12_1 = flow['flow_12_1']  # B, 2, W, H
        warp_ref = self.warp(ref, flow_12_1)
        return warp_ref, flow_12_1

    def stage2_align(self, x, ref):
        flow = self.flownet2(x, ref)
        flow_12_1 = flow['flow_12_1']
        flow_12_2 = flow['flow_12_2']
        flow_12_3 = flow['flow_12_3']
        flow_12_4 = flow['flow_12_4']
        ref_conv1, ref_conv2, ref_conv3, ref_conv4 = self.encoder(ref)
        warp_21_conv1 = self.warp(ref_conv1, flow_12_1)
        warp_21_conv2 = self.warp(ref_conv2, flow_12_2)
        warp_21_conv3 = self.warp(ref_conv3, flow_12_3)
        warp_21_conv4 = self.warp(ref_conv4, flow_12_4)
        return warp_21_conv1, warp_21_conv2, warp_21_conv3, warp_21_conv4