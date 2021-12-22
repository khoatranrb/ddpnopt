import numpy as np
import torch.nn as nn
import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
from collections import namedtuple

def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__

class Step(nn.Module):
    def __init__(self, module):
        super(Step, self).__init__()
        self.x = None
        self.x_old = None
        self.mod = module
        self.type = _layer_type(module)
    def __str__(self):
        return 'DDP'
    def forward(self, inp=None, update=False, opt=None):
        if not update:
            self.x_old = inp
            self.x = nn.Parameter(inp)
            return (self.mod(self.x)+self.mod(inp))/2
        else:
            try: return self.update(inp, opt)
            except: return self.update(self.x, opt)

    def forward_wo_train(self, inp):
        return self.mod(inp)

    @torch.no_grad()
    def update(self, inp, opt):
        Q_u = self.mod.weight.grad
        Q_uu = opt.state[self.mod.weight]['hess']
        Q_x = self.x.grad.mean(dim=0)
        if self.type=='Linear':
            Q_ux = calc_q_ux_fc(Q_u, Q_x.unsqueeze(-1))
            big_k = calc_big_k_fc(Q_uu, Q_ux)
            term2 = torch.einsum('xy,zt->xt', big_k, (inp - self.x_old).mean(dim=0).unsqueeze(-1)).squeeze(-1).reshape(
                self.mod.weight.shape)
        else:
            Q_ux = calc_q_ux_conv(self.mod.weight, Q_u)
            big_k = calc_big_k_conv(Q_uu, Q_ux)
            term2 = calc_term2_conv(big_k, inp - self.x_old).reshape(self.mod.weight.shape)
        self.mod.weight += opt.lr*opt.lrddp*(term2)
        out = self.forward_wo_train(inp)
        del self.x
        del self.x_old
        del opt.state[self.mod.weight]['hess']
        return out


def calc_q_ux_fc(q_u, q_x):
    return torch.einsum('xy,zt->xt', q_u.flatten(start_dim=0).unsqueeze(-1), torch.transpose(q_x,0,1))

def calc_q_ux_conv(W, q_u):
    return torch.einsum('cdef,cdef->cdef',W,q_u)

def calc_small_k(q_uu, q_u):
    return -q_u/q_uu

def calc_big_k_conv(q_uu, q_ux):
    return -q_ux/q_uu
        
def calc_big_k_fc(q_uu, q_ux):
    return -torch.einsum('x,xt->xt', 1/q_uu.flatten(start_dim=0), q_ux)

def calc_term2_conv(big_k, delta_x):
    conv_size = big_k.shape[-1]
    b, c, h, w = delta_x.shape
    t = torch.as_strided(delta_x, size=(b, h - conv_size+1, w - conv_size+1, c, conv_size,conv_size), stride=(c * h * h, w, 1, h * w, w, 1)).mean(dim=0)
    t = t.flatten(start_dim=0, end_dim=1).flatten(start_dim=-2, end_dim=-1)
    H = t.shape[0]
    big_k = big_k.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True).repeat((1,1,1,H))
    out = torch.einsum('ncxh,hcd->ncdx', big_k, t)
    return out
  
class MySeq(nn.Sequential):
    def __str__(self): return 'DDP'
    def forward(self, a, update=None, opt=None):
        for module in self:
            if str(module) in ['DDP']:
                a = module(a, update, opt)
            else: a = module(a)
        return a

class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = Step(nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False))
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
        return MySeq(*layers)
        return nn.Sequential(*layers)
        
    def forward(self, x=None, update=False, opt=None):
        
        x = self.conv1(x, update, opt)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, update, opt)
        x = self.layer2(x, update, opt)
        x = self.layer3(x, update, opt)
        x = self.layer4(x, update, opt)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x
      
class Bottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
    
        self.conv1 = Step(nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                               stride = 1, bias = False))
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = Step(nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False))
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = Step(nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False))
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = Step(nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                             stride = stride, bias = False))
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
            
        self.downsample = downsample
        
    def forward(self, x, update=None, opt=None):
        
        i = x
        
        x = self.conv1(x, update, opt)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x, update, opt)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x, update, opt)
        x = self.bn3(x)
                
        if self.downsample is not None:
            i = self.downsample(i)
            
        x += i
        x = self.relu(x)
    
        return x
    
def get_cfg():
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    return ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 8, 36, 3],
                                channels = [64, 128, 256, 512])
