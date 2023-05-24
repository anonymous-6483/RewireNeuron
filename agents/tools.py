#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from .softsort import softsort
import numpy as np


def create_dist(dist_type,n_anchors):
    n_anchors = max(1,n_anchors)
    if dist_type == "flat":
        dist = Dirichlet(torch.ones(n_anchors))
    if dist_type == "peaked":
        dist = Dirichlet(torch.Tensor([1.] * (n_anchors-1) + [n_anchors ** 2]))
    elif dist_type == "categorical":
        dist = Categorical(torch.ones(n_anchors))
    elif dist_type == "last_anchor":
        dist = Categorical(torch.Tensor([0] * (n_anchors-1) + [1]))
    return dist


class LinearSubspace(nn.Module):
    def __init__(self, n_anchors, in_channels, out_channels, bias = True, same_init = False, freeze_anchors = True):
        super().__init__()
        self.n_anchors = n_anchors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias
        self.freeze_anchors = freeze_anchors

        if same_init:
            anchor = nn.Linear(in_channels,out_channels,bias=self.is_bias)
            anchors = [copy.deepcopy(anchor) for _ in range(n_anchors)]
        else:
            anchors = [nn.Linear(in_channels,out_channels,bias=self.is_bias) for _ in range(n_anchors)]
        self.anchors = nn.ModuleList(anchors)

    def forward(self, x, alpha):
        #print("---anchor:",max(x.abs().max() for x in self.anchors.parameters()))
        #check = (not torch.is_grad_enabled()) and (alpha[0].max() == 1.)
        xs = [anchor(x) for anchor in self.anchors]
        #if check:
        #    copy_xs = xs
        #    argmax = alpha[0].argmax()
        xs = torch.stack(xs,dim=-1)

        alpha = torch.stack([alpha] * self.out_channels, dim=-2)
        xs = (xs * alpha).sum(-1)
        #if check:
        #    print("sanity check:",(copy_xs[argmax] - xs).sum().item())
        return xs

    def add_anchor(self,alpha = None):
        if self.freeze_anchors:
            for param in self.parameters():
                param.requires_grad = False

        # Midpoint by default
        if alpha is None:
            alpha = torch.ones((self.n_anchors,)) / self.n_anchors

        new_anchor = nn.Linear(self.in_channels,self.out_channels,bias=self.is_bias)
        new_weight = torch.stack([a * anchor.weight.data for a,anchor in zip(alpha,self.anchors)], dim = 0).sum(0)
        new_anchor.weight.data.copy_(new_weight)
        if self.is_bias:
            new_bias = torch.stack([a * anchor.bias.data for a,anchor in zip(alpha,self.anchors)], dim = 0).sum(0)
            new_anchor.bias.data.copy_(new_bias)
        self.anchors.append(new_anchor)
        self.n_anchors +=1


class Sequential(nn.Sequential):
    def forward(self, input, alpha):
        for module in self:
            input = module(input,alpha) if isinstance(module,LinearSubspace) else module(input)
        return input


def init_sort(v=None, n=None, k=None, scale=1.):
    if v is None:
        if k is None:
            k = 1
        v = torch.stack([torch.arange(n) for _ in range(k)]).float()
    else:
        v = v.argsort(dim=1).argsort(dim=1).float()
    return (v - v.min(dim=1)[0][:, np.newaxis]) / (v.max(dim=1)[0] - v.min(dim=1)[0])[:, np.newaxis] * scale


def mul_grad(x, k):
    return x * k - x.detach() * (k - 1)


class LinearRewire(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, tau=1.0, beta=1.0, k=3, tau2=1.0, beta2=1.0, cycle=-1, note=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias
        self.linear = nn.Linear(in_channels, out_channels, bias=self.is_bias)

        self.tau = tau
        self.beta = beta
        self.k = k
        self.tau2 = tau2
        self.beta2 = beta2
        self.cycle = cycle
        self.note = note  # 1 post, 2 pre, 3 both
        if self.note != 2:
            self.v = nn.Parameter(init_sort(n=out_channels, k=k))

        self.cnt = 0
        self.t = None

    def forward(self, x, is_train=False, k=0):
        out = self.linear(x)
        if self.note == 2:
            return out
        elif self.t is None:
            if is_train:
                p = softsort(mul_grad(self.v[k], self.k), tau=self.tau, beta=self.beta)
                return out @ p.T
            else:
                return out[..., self.v[k].argsort()]
        else:
            if self.cycle == -1:
                t = min(self.t, self.cnt - 1)
            else:
                t = min(self.t % self.cycle, self.cnt - 1)
            return out[..., self.get_buffer(f"v{t}v")]

    def set_task(self, task_id=None, k=0):
        if self.note == 2:
            if task_id == -1:
                self.cnt += 1
        elif task_id == -1:
            cnt = self.cnt
            if self.cycle != -1:
                cnt = cnt % self.cycle
            self.register_buffer(f"v{cnt}v", self.v[k].argsort())
            self.v.data = torch.stack([self.v[k].data.clone() for _ in range(self.k)])
            self.cnt += 1
        else:
            self.t = task_id
            if task_id is None:
                if (self.cycle != -1) and (self.cnt >= self.cycle):
                    last_v = self.get_buffer(f"v{self.cnt % self.cycle}v").argsort()
                    self.v.data = torch.stack([last_v.clone() for _ in range(self.k)])
                self.v.data = init_sort(v=self.v.data)

    def pre_register_and_consolidate(self):
        av, zv = None, None
        if self.cnt > 0:
            if self.note // 2 == 1:
                av = self.av.argsort().argsort()
            if self.note % 2 == 1:
                zv = self.zv.argsort()
        return av, zv

    def register_and_consolidate(self, zv, next_av):
        # update self.vnv
        if (self.cnt > 0) and (self.note != 2):
            cnt = self.cnt
            if self.cycle != -1:
                cnt = min(cnt, self.cycle)
            for t in range(cnt):
                if next_av is not None:
                    self.register_buffer(f"v{t}v", zv[self.get_buffer(f"v{t}v")][next_av])
                else:
                    self.register_buffer(f"v{t}v", zv[self.get_buffer(f"v{t}v")])
        # register mean
        for name, param in self.named_parameters():
            if name.endswith('v'): continue
            name = name.replace('.', '_')
            self.register_buffer(f"{name}_mean", param.data.clone())
        # initialize self.vv
        if self.note // 2 == 1:
            self.register_parameter('av', nn.Parameter(init_sort(n=self.in_channels)[0]))
        if self.note % 2 == 1:
            self.register_parameter('zv', nn.Parameter(init_sort(n=self.out_channels)[0]))

    def add_regularizer(self):
        losses = []
        if self.note // 2 == 1:
            ap = softsort(self.av, tau=self.tau2, beta=self.beta2)
        if self.note % 2 == 1:
            zp = softsort(self.zv, tau=self.tau2, beta=self.beta2)
        for name, param in self.named_parameters():
            if name.endswith('v'): continue
            name = name.replace('.', '_')
            mean = self.get_buffer(f"{name}_mean")
            if (self.note == 1) or ((self.note == 3) and name.endswith('bias')):
                losses.append(- 2 * ((zp.T @ mean) * param).sum() + (param ** 2).sum())
            elif (self.note == 2) and name.endswith('weight'):
                losses.append(- 2 * ((mean @ ap) * param).sum() + (param ** 2).sum())
            elif (self.note == 3) and name.endswith('weight'):
                losses.append(- 2 * ((zp.T @ mean @ ap) * param).sum() + (param ** 2).sum())
            else:  # self.note == 2 and name.endswith('bias')
                losses.append(((mean - param) ** 2).sum())
        return losses


class LinearExpand(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, cycle=-1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias
        self.cycle = cycle
        self.linears = nn.ModuleList([])

        self.cnt = 0
        self.t = None

    def forward(self, x):
        if self.t is None:
            t = self.cnt
            if self.cycle != -1:
                t = t % self.cycle
        else:
            if self.cycle == -1:
                t = min(self.t, self.cnt - 1)
            else:
                t = min(self.t % self.cycle, self.cnt - 1)
        return self.linears[t](x)

    def set_task(self, task_id=None):
        if task_id == -1:
            self.cnt += 1
        else:
            self.t = task_id
            if task_id is None:
                if self.cnt == 0:
                    self.linears.append(nn.Linear(self.in_channels, self.out_channels, bias=self.is_bias).cuda())
                elif (self.cycle == -1) or (self.cnt < self.cycle):
                    self.linears.append(copy.deepcopy(self.linears[-1]))
                    for param in self.linears[-2].parameters():
                        param.requires_grad = False
                else:
                    for param in self.linears[self.cnt % self.cycle].parameters():
                        param.requires_grad = True
                    for param in self.linears[(self.cnt - 1) % self.cycle].parameters():
                        param.requires_grad = False


class SequentialRewire(nn.Sequential):
    def forward(self, input, is_train=False, k=0):
        for module in self:
            input = module(input, is_train, k) if isinstance(module, LinearRewire) else module(input)
        return input

    def set_task(self, task_id=None, k=0):
        for module in self:
            if isinstance(module, LinearRewire):
                module.set_task(task_id, k)
            elif isinstance(module, LinearExpand):
                module.set_task(task_id)
