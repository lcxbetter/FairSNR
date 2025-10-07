import random

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import Dataset
from typing import Callable



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # Make sure numpy is imported if used elsewhere



import torch
import torch.nn as nn
import torch.nn.functional as F


class SNLoss(nn.Module):
    def __init__(self, num_classes, int_lambda=0.05, int_reg=0.001, target_lambda=0.2, kl_lambda=0.001, bias=1,
                 prior_type='conditional'):
        super().__init__()
        self.num_classes = num_classes
        self.int_lambda = int_lambda
        self.int_reg = int_reg
        self.target_lambda = target_lambda
        self.kl_lambda = kl_lambda
        self.bias = bias
        self.prior_type = prior_type
        self.mse = nn.MSELoss()

    def kl_normal(self, qm, qv, pm, pv):
        # qm, pm: (N, dim)
        # qv, pv: (N, dim)
        return ((qm - pm) ** 2 / (pv + 1e-6)).mean()  # 可选加权 KL

    def condition_prior(self, scale, label, dim, device):
        mean = ((label - scale[0]) / (scale[1] - scale[0])).reshape(-1, 1).repeat(1, dim)
        var = torch.ones(label.size(0), dim, device=device)
        return mean, var

    def kl_loss(self, m, v, y):
        # m, v: (N, dim)
        device = m.device
        if self.prior_type == 'conditional':
            pm, pv = self.condition_prior([0, self.num_classes], y, m.size(1), device)
        else:
            pm, pv = torch.zeros_like(m), torch.ones_like(m)
        return self.kl_normal(m, v * 0.0001, pm, pv * 0.0001)

    def intervention_loss(self, intervention):
        return torch.norm(torch.pow(intervention, 2) - self.bias)

    def targets_loss(self, y_pred, int_y_pred):
        return -self.mse(torch.sigmoid(y_pred), torch.sigmoid(int_y_pred))

    def forward(self, z, int_z, m, v, y, y_pred, int_y_pred, z_c, train_mask, turn='min'):

        z_train = z[train_mask]
        int_z_train = int_z[train_mask]
        m_train = m[train_mask]
        v_train = v[train_mask]
        intervention_train = z_c[train_mask]
        y_float = y.float()
        int_m_train = int_z_train
        int_v_train = torch.zeros_like(int_z_train)
        nll = F.binary_cross_entropy_with_logits(y_pred, y_float).mean()
        int_nll = -F.binary_cross_entropy_with_logits(int_y_pred, y_float).mean()
        inter_norm = self.intervention_loss(intervention_train)
        targets_loss = self.targets_loss(y_pred, int_y_pred).mean()
        # KL loss: 用m_train, v_train, y
        kl = self.kl_loss(m_train, v_train, y).mean() + self.kl_loss(int_m_train, int_v_train, y).mean()


        all_loss = nll + self.int_lambda * int_nll + self.int_reg * inter_norm + self.target_lambda * targets_loss
        if turn == 'min':
            return all_loss + self.kl_lambda * kl
        else:
            return -all_loss + self.kl_lambda * kl



