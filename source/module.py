from functools import partial
import torch.nn.functional as F
import torch.optim as optim
import torch
import flow_vis

import torchlight
from torchlight.nn.loss import charbonnier_loss
from torchlight.utils.helper import get_obj
from torchlight.utils.metrics import psnr

from .model import EnhancedMultiscaleWarpingNet, MultiscaleWarpingNet
from .loss import feature_warping_loss, landmark_loss


class LightField_RefSR_Module(torchlight.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.optimizer = get_obj(optimizer, optim, self.model.parameters())
        self.metric_ftns = [psnr]

    def step(self, data, train, epoch, step):
        if train:
            self.optimizer.zero_grad()
        data, matches = data
        matches = [m.float().to(self.device) for m in matches]
        data = (d.to(self.device) for d in data)
        loss, result = self._step((data, matches), train, epoch, step)
        if train:
            loss.backward()
            self.optimizer.step()
        return result

    def _step(self, data, train, epoch, step):
        raise NotImplementedError

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])


class CrossNet_Module(LightField_RefSR_Module):
    def __init__(self, optimizer):
        super().__init__(MultiscaleWarpingNet(), optimizer)

    def _step(self, data, train, epoch, step):
        data, matches = data
        img1_LR, img1_HR, img1_SR, img2_HR, img2_LR = data
        target = img1_HR
        input = {
            'input_img1_LR': img1_LR,
            'input_img1_SR': img1_SR,
            'input_img2_HR': img2_HR,
            'input_img2_LR': img2_LR
        }

        output = self.model(input)
        reconstruct_loss = charbonnier_loss(output, target)

        loss = reconstruct_loss

        metrics = {'loss': loss.item(), 'r_loss': reconstruct_loss.item()}

        for met in self.metric_ftns:
            metrics[met.__name__] = met(output, target)
            metrics[met.__name__ + '_ref'] = met(img2_HR, target)

        imgs = {'combine': [img1_LR[0], img1_SR[0], output[0], img1_HR[0]]}

        return loss, self.StepResult(imgs=imgs, metrics=metrics)


class ConfigurableLossCalculator:
    def __init__(self, cfg:dict):
        self.cfg = cfg
        self.loss_fns = {}
    
    def register(self, fn, name):
        self.loss_fns[name] = fn
    
    def compute(self):
        loss = 0
        loss_dict = {}
        for name, weight in self.cfg.items():
            l = self.loss_fns[name]() * weight
            loss_dict[name] = l.item()
            loss += l
        loss_dict['total'] = loss.item()
        return loss, loss_dict

class CrossNet_PP_Module(LightField_RefSR_Module):
    def __init__(self, optimizer, loss_config={'reconstruct': 1}, mode='input_img1_SR'):
        super().__init__(EnhancedMultiscaleWarpingNet(), optimizer)
        self.loss_config = loss_config
        self.loss_calculator = ConfigurableLossCalculator(loss_config['weight'])
        
        self.reconstruct_loss = partial(charbonnier_loss, reduce=loss_config['re_reduce'])
        self.mode = mode
        
    def _step(self, data, train, epoch, step):
        data, matches = data
        img1_LR, img1_HR, img1_SR, img2_HR, img2_LR = data
        target = img1_HR
        input = {
            'input_img1_LR': img1_LR,
            'input_img1_SR': img1_SR,
            'input_img1_HR': img1_HR,
            'input_img2_HR': img2_HR,
            'input_img2_LR': img2_LR
        }

        output, warped, flow = self.model(input, mode=self.mode)
        warped_ref, warped_feats = warped
   
        self.loss_calculator.register(lambda : self.reconstruct_loss(output, target), 'reconstruct')
        self.loss_calculator.register(lambda : landmark_loss(matches, flow), 'landmark')
        self.loss_calculator.register(lambda : F.mse_loss(warped_ref, target), 'warp')
        self.loss_calculator.register(lambda : feature_warping_loss(warped_feats, target, self.model.encoder), 
                                      name='feat_warp')
        
        loss, metrics = self.loss_calculator.compute()
          
        for met in self.metric_ftns:
            metrics[met.__name__] = met(output, target)
            metrics[met.__name__ + '_warp'] = met(warped_ref, target)
            metrics[met.__name__ + '_ref'] = met(img2_HR, target)

        flow_color = flow[0].detach().cpu().numpy().transpose(1, 2, 0)
        flow_color = flow_vis.flow_to_color(flow_color, convert_to_bgr=False)
        flow_color = torch.from_numpy(flow_color.transpose(2, 0, 1)).float().to(flow.device) / 255
        warp_diff = torch.abs(warped_ref[0]-target[0])
        pred_diff = torch.abs(output[0]-target[0])
        
        imgs = {'combine': [img1_LR[0], img1_SR[0], output[0], img1_HR[0]],
                'ref': [warped_ref[0], img2_HR[0], flow_color],
                'diff': [warp_diff, pred_diff],
                }
        return loss, self.StepResult(imgs=imgs, metrics=metrics)