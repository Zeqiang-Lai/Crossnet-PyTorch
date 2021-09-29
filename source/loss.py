import torch.nn.functional as F
import torchlight.nn.loss as tl

def landmark_loss(landmarks, flow):
    """ 
        landmarks: list of cords of each paired landmarks [B] -> [N,4]
        flow: offset of the flow [B,2,W,H]
    """
    loss = 0
    for idx, lm in enumerate(landmarks):
        source_lm = lm[:,2:4]
        target_lm = lm[:,0:2]
        loss += tl.landmark_loss(source_lm.unsqueeze(0), target_lm.unsqueeze(0), flow[idx:idx+1])
    return loss / len(landmarks)
