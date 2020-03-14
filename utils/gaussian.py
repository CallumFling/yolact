# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import _batch_mahalanobis
from data.config import cfg, mask_type
import torchsnooper
import snoop

torchsnooper.register_snoop()


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


def rotation_matrix(theta):
    # Theta shape: torch.shape(batch,num_priors)
    sin, cos = torch.sin(theta), torch.cos(theta)

    # shape: torch.shape(batch,num_priors,2)
    rot_0 = torch.stack((cos, -sin), dim=2)
    rot_1 = torch.stack((sin, cos), dim=2)

    # shape: torch.shape(batch,num_priors,2,2)
    rot = torch.stack((rot_0, rot_1), dim=2)
    return rot


# @snoop
def gauss_loc(loc):
    # Σ is equal to RSS(R-1)
    # Cholesky=T=RS
    # calculate this instead
    # loc shape: torch.size(batch_size,num_priors,5)
    locShape = list(loc.shape)

    mean = loc[:, :, :2]
    # mean = torch.tanh(mean)
    mean = torch.fmod(mean, cfg.gauss_wrap)

    theta = loc[:, :, 2]
    rot = rotation_matrix(theta)

    scale_coeff = loc[:, :, 3:5]
    # Positive required
    scale_coeff = (
        (torch.sign(scale_coeff) * cfg.max_scale)
        - torch.fmod(scale_coeff, cfg.max_scale)
    ) ** 2 + cfg.positive
    scale = torch.diag_embed(scale_coeff)
    cholesky = rot @ scale
    return mean, cholesky


# @snoop
def mahalanobisGrid(predictions=None, maskShape=None, loc=None):
    if maskShape is None:
        maskShape = list(predictions["proto"].shape)[1:3]
    if loc is None:
        loc = predictions["loc"]

    locShape = list(loc.shape)[:-1]
    mean, cholesky = gauss_loc(loc)

    # Resize to Batch Muahalanobis Specs
    cholesky = cholesky.view(-1, 1, 2, 2)
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
        mahalanobis = -0.5 * _batch_mahalanobis(cholesky.float(), diff.float()).half()
    else:
        mahalanobis = -0.5 * _batch_mahalanobis(cholesky, diff)

    # Dim: Batch, Anchors, i, j
    result = mahalanobis.view(*locShape, *maskShape)
    if torch.isnan(mahalanobis).any():
        __import__("pdb").set_trace()

    return result


def unnormalGaussian(predictions=None, maskShape=None, loc=None):
    result = torch.pow(
        torch.tensor([math.e]),
        mahalanobisGrid(predictions=predictions, maskShape=maskShape, loc=loc),
    )
    if cfg.use_amp:
        return result.half()
    return result
