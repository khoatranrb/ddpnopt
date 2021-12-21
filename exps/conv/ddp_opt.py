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
                
class AdamDDP(Optimizer):
    def __init__(self, model, lr=1e-3, lrddp = 1e-2, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, maximize=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super(AdamDDP, self).__init__(model.parameters(), defaults)
        self.lr = lr
        self.lrddp = lrddp
        self.model = model
        self.eps = eps
        self.beta1 = betas[0]
        self.beta2 = betas[1]

    def __setstate__(self, state):
        super(AdamDDP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1

            self.update()
            if self.lrddp: self.model(update=True, opt=self)

    def update(self):
      for group in self.param_groups:
          for p in group['params']:
              if p.grad is not None:
                  if p.grad.is_sparse:
                      raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                  state = self.state[p]
                  bias_correction1 = 1 - self.beta1 ** state['step']
                  bias_correction2 = 1 - self.beta2 ** state['step']
                  
                  state['exp_avg'].mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
                  state['exp_avg_sq'].mul_(self.beta2).addcmul_(p.grad, p.grad.conj(), value=1 - self.beta2)
                  
                  denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                  state['hess'] = denom
                  step_size = self.lr / bias_correction1
                  
                  p.addcdiv_(state['exp_avg'], denom, value=-step_size)
