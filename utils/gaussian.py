# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import _batch_mahalanobis
from data.config import cfg, mask_type


# class paramaterActivation(nn.Module):
# def __init__(self):
# super(paramaterActivation, self).__init__()
# self.identity = lambda x: x
# # mean x mean y cov xx cov xy cov yy
# self.activations = [self.identity, self.identity, F.relu, self.identity, F.relu]

# def forward(self, tensor):
# result = [f(tensor[:, :, i]) for i, f in enumerate(self.activations)]
# # unsup: Be careful with this dimension, because input is N, C, H, W
# return torch.stack(result, dim=-1)


# Pad one left, don't pad right, fill with 0
left_pad = torch.nn.ConstantPad1d((1, 0), 0)


def lincomb(predictions=None, proto=None, masks=None):
    # proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)
    # masks shape: torch.size(batch_size,num_priors,mask_dim)
    if proto is None:
        proto = predictions["proto"]
    if masks is None:
        masks = predictions["masks"]

    # unsup: optimize einsum
    # Dim: Batch, #Priors, H, W
    return torch.einsum("acde,abe->abcd", proto, masks)


# def flipDiagonal(x):
# # https://stackoverflow.com/questions/16444930/copy-upper-triangle-to-lower-triangle-in-a-python-matrix
# print("x shape", x.shape)
# return (
# x + x.transpose(-1, -2) - torch.diag_embed(torch.diagonal(x, dim1=-1, dim2=-2))
# )


def gauss_loc(loc):
    # loc shape: torch.size(batch_size,num_priors,6)
    locShape = list(loc.shape)
    mean = loc[:, :, :2]
    cov = loc[:, :, 2:].view(*locShape[:2], 2, 2)
    cov = (cov.permute(0, 1, 3, 2)) @ cov
    # cov = cov + torch.eye(2) * cfg.positive
    cov = torch.abs(cov)
    return mean, cov


def mahalanobisGrid(predictions=None, maskShape=None, loc=None):
    """

    mean shape ()
    """
    if maskShape is None:
        maskShape = list(predictions["proto"].shape)[1:3]
    if loc is None:
        loc = predictions["loc"]

    locShape = list(loc.shape)[:-1]
    # batch,priors,2
    # cov = (loc[:, :, 2:4], left_pad(loc[:, :, 4:]))
    # cov = flipDiagonal(torch.stack(cov, dim=2))

    mean, cov = gauss_loc(loc)
    # Resize to Batch Muahalanobis Specs
    cov = cov.view(-1, 1, 2, 2)
    mean = mean.view(-1, 1, 2)

    j, i = torch.meshgrid(
        torch.linspace(-1, 1, maskShape[0]), torch.linspace(-1, 1, maskShape[1])
    )
    i, j = i.contiguous(), j.contiguous()
    # unsup: Cache this to optimize
    # becomes (0,0),(1,0),(2,0)...
    coordinate_list = torch.stack((i.view(-1), j.view(-1)), dim=1)
    diff = coordinate_list - mean

    # this is Squared Mahalanobis
    if cfg.use_amp:
        mahalanobis = -0.5 * _batch_mahalanobis(cov.float(), diff.float()).half()
    else:
        mahalanobis = -0.5 * _batch_mahalanobis(cov, diff)

    # Dim: Batch, Anchors, i, j
    return mahalanobis.view(*locShape, *maskShape)


def unnormalGaussian(predictions=None, maskShape=None, loc=None):
    result = torch.pow(
        torch.tensor([math.e]),
        mahalanobisGrid(predictions=predictions, maskShape=maskShape, loc=loc),
    )
    if cfg.use_amp:
        return result.half()
    return result
