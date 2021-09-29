from functools import partial
import torch.nn.functional as F
import torch.optim as optim
import torch
import flow_vis

import torchlight
from torchlight.nn.loss import charbonnier_loss
from torchlight.utils.helper import get_obj, to_device, ConfigurableLossCalculator
from torchlight.utils.metrics import psnr

from .model import EnhancedMultiscaleWarpingNet, MultiscaleWarpingNet
from .loss import landmark_loss


class BaseModule(torchlight.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        self.optimizer = get_obj(optimizer, optim, self.model.parameters())
        self.metric_ftns = [psnr]

    def step(self, data, train, epoch, step):
        self.model.eval()
        if train:
            self.optimizer.zero_grad()
            self.model.train()
        loss, result = self._step(data, train, epoch, step)
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


class CrossNetModule(BaseModule):
    def __init__(self, optimizer, input_mode):
        super().__init__(MultiscaleWarpingNet(), optimizer)
        self.input_mode = input_mode

    def _step(self, data, train, epoch, step):
        data = to_device(data, self.device)

        img1_SR = data['img1_'+self.input_mode['input']]
        img2_HR = data['img2_HR']
        img1_flow = data['img1_'+self.input_mode['flow'][0]]
        img2_flow = data['img2_'+self.input_mode['flow'][1]]
        target = data['img1_HR']

        output = self.model(img1_SR, img2_HR, img1_flow, img2_flow)

        loss = charbonnier_loss(output, target, reduce='batch')

        metrics = {'loss': loss.item()}
        for met in self.metric_ftns:
            metrics[met.__name__] = met(output, target)
            metrics[met.__name__ + '_ref'] = met(img2_HR, target)

        imgs = {'combine': [img1_SR[0], output[0], target[0]]}

        return loss, self.StepResult(imgs=imgs, metrics=metrics)


class EnhancedCrossNetModule(BaseModule):
    def __init__(self, optimizer, input_mode, loss_config={'reconstruct': 1}):
        super().__init__(EnhancedMultiscaleWarpingNet(), optimizer)
        self.input_mode = input_mode

        self.loss_config = loss_config
        self.loss_calculator = ConfigurableLossCalculator(loss_config['weight'])

        self.reconstruct_loss = partial(charbonnier_loss, reduce=loss_config['reduce'])

    def _step(self, data, train, epoch, step):
        data = to_device(data, self.device)

        # ------------------------------- prepare data ------------------------------- #

        img1_SR = data['img1_'+self.input_mode['input']]
        img2_HR = data['img2_HR']
        target = data['img1_HR']
        landmark = data['landmark']

        # ---------------------------------- forward --------------------------------- #

        output, warped, flow = self.model(img1_SR, img2_HR)
        warped_ref, _ = warped

       # ------------------------------- compute loss ------------------------------- #

        self.loss_calculator.register(lambda: self.reconstruct_loss(output, target), 'reconstruct')
        self.loss_calculator.register(lambda: landmark_loss(landmark, flow), 'landmark')
        self.loss_calculator.register(lambda: F.mse_loss(warped_ref, target), 'warp')

        total_loss, metrics = self.loss_calculator.compute()
        metrics['total'] = total_loss.item()
        
        # ------------------------------ compute metrics ----------------------------- #
        
        for met in self.metric_ftns:
            metrics[met.__name__] = met(output, target)
            metrics[met.__name__ + '_warp'] = met(warped_ref, target)
            metrics[met.__name__ + '_ref'] = met(img2_HR, target)

        # ------------------------------- visualization ------------------------------ #

        flow_color = flow[0].detach().cpu().numpy().transpose(1, 2, 0)
        flow_color = flow_vis.flow_to_color(flow_color, convert_to_bgr=False)
        flow_color = torch.from_numpy(flow_color.transpose(2, 0, 1)).float().to(flow.device) / 255
        warp_diff = torch.abs(warped_ref[0]-target[0])
        pred_diff = torch.abs(output[0]-target[0])

        imgs = {'combine': [img1_SR[0], output[0], target[0]],
                'ref': [warped_ref[0], img2_HR[0], flow_color],
                'diff': [warp_diff, pred_diff],
                }

        return total_loss, self.StepResult(imgs=imgs, metrics=metrics)
