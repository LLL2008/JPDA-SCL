from __future__ import print_function
import torch
import torch.nn as nn
from torch import linalg as LA
import numpy as np


class RCS(nn.Module):
    def __init__(self):
        super(RCS,self).__init__()

    def forward(self,FX,y,l,alpha,sigma,lamda):
        m = FX.shape[0]    
        Deltay = torch.as_tensor(y[:,None]==y,dtype=torch.float64,device=FX.device)
        Deltal = torch.as_tensor(l[:,None]==l,dtype=torch.float64,device=FX.device)
        FX_norm = torch.sum(FX ** 2, axis = -1)
        K = torch.exp(-(FX_norm[:,None] + FX_norm[None,:] - 2 * torch.matmul(FX, FX.t())) / sigma) * Deltay
                        
        P = K * Deltal
        H = ((1.0 - alpha) / m**2) * torch.matmul(K,K) * torch.matmul(Deltal,Deltal) + (1.0 * alpha / m) * torch.matmul(P,P)
        h = torch.mean(P,axis=0)
        theta = torch.linalg.solve(H + lamda * torch.eye(m, device=FX.device), h)
        D = 2 * torch.matmul(h,theta) - torch.matmul(theta,torch.matmul(H,theta)) - 1
        return D

def Rloss(side_1_1, side_2_1, la1, la2):

    rcs = RCS()
    source_features1 = torch.cat((side_1_1.view(side_1_1.shape[0], -1),
                                  side_2_1.view(side_2_1.shape[0], -1)), dim=0)
    lab = torch.cat((la1, la2), dim=0)
    l = torch.cat((torch.ones((len(la1), 1)),
                   2 * torch.ones((len(la2), 1))), dim=0)
    y = torch.argmax(lab, dim=1).squeeze().cuda()
    l = l.squeeze().cuda()

    pairwise_dist = torch.cdist(source_features1, source_features1, p=2) ** 2
    sigma = torch.median(pairwise_dist[pairwise_dist > 0])

    Rcs_loss = rcs(source_features1.double(), y, l, 0.5, sigma, 1e-3)
    return Rcs_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

        self.base_temperature = temperature
        # self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss