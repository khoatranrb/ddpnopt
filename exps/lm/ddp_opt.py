import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import List, Optional

                        
class RMS_DDP(Optimizer):
    def __init__(self, model, lr=1e-2, lrddp = 1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        self.lr = lr
        self.lrddp = lrddp
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.first_time = True
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMS_DDP, self).__init__(model.parameters(), defaults)
        for p in self.model.parameters():
            self.state[p]['feedback'] = False

    def __setstate__(self, state):
        super(RMS_DDP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, src, mask):
        if self.first_time:
            self.model(src, mask, update=True, opt=self)
            self.first_time = False
        list_params, list_grad = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or self.state[p]['feedback']:
                    continue
                state = self.state[p]
                if len(state) < 2:
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                list_params.append(p)
                list_grad.append(p.grad)
        self.update(list_params, list_grad)
        if self.lrddp: self.model(src, mask, update=True, opt=self)

    def update(self, list_params, list_grad):
        for p, grad in zip(list_params, list_grad):
            state = self.state[p]
            state['square_avg'].mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
            avg = state['square_avg'].sqrt().add_(self.eps)
            p.addcdiv_(grad, avg, value=-self.lr)
