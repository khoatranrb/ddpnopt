import numpy as np
import torch.nn as nn
import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
import math


class DDPNOPT(Optimizer):
    def __init__(self, model, lr=1e-2, lrddp = 1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        self.lr = lr
        self.lrddp = lrddp
        self.model = model
        self.eps = eps
        self.alpha = alpha
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(DDPNOPT, self).__init__(model.parameters(), defaults)

    def __setstate__(self, state):
        super(DDPNOPT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['step'] += 1


        self.update()
        if self.lrddp: self.model(update=True, opt=self)

    def update(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['square_avg'].mul_(self.alpha).addcmul_(p.grad, p.grad, value=1 - self.alpha)
                avg = state['square_avg'].sqrt().add_(self.eps)
                state['hess'] = avg
                p.addcdiv_(p.grad, avg, value=-self.lr)
