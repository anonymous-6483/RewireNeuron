import torch
import torch.nn.functional as F
import random


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def softsort(
    scores,
    tau=1.0,
    beta=0.0
):
    scores = scores + sample_gumbel(scores.shape).to(scores.device) * beta
    scores = scores.unsqueeze(-1)
    sorted = scores.sort(dim=-2)[0]
    pairwise_diff = (scores.transpose(-2, -1) - sorted).abs().neg()
    soft = torch.softmax(pairwise_diff / tau, dim=-1)
    hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1)
    return hard + soft - soft.detach()
