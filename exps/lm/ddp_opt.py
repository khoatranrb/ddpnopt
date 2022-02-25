import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import List, Optional
import math

class RmsDDP(Optimizer):
    def __init__(self, model, lr=1e-2, lrddp = 1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        self.lr = lr
        self.lrddp = lrddp
        self.model = model
        self.eps = eps
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RmsDDP, self).__init__(model.parameters(), defaults)

    def __setstate__(self, state):
        super(RmsDDP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)
            
    def update(self, params, grads, square_avgs):
        for i, param in enumerate(params):
            grad = grads[i]
            square_avg = square_avgs[i]

            square_avg.mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)

            avg = square_avg.sqrt().add_(self.eps)

            param.addcdiv_(grad, avg, value=-self.lr)

    @torch.no_grad()
    def step(self, src, mask):
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])



        self.update(params_with_grad,
                  grads,
                  square_avgs)
        if self.lrddp: self.model(src, mask, update=True, opt=self)
