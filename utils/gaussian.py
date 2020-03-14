# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import _batch_mahalanobis
from data.config import cfg, mask_type


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


def gauss_loc(loc):
    # Σ is equal to RSS(R-1)
    # Cholesky=T=RS
    # calculate this instead
    # loc shape: torch.size(batch_size,num_priors,6)
    locShape = list(loc.shape)

    # NOTE: without tanh, something becomes NaN
    loc = torch.tanh(loc)
    mean = loc[:, :, :2]
    cov = loc[:, :, 2:].view(*locShape[:2], 2, 2)
    cov = (cov.permute(0, 1, 3, 2)) @ cov
    # cov = cov + torch.eye(2) * cfg.positive
    # NOTE: this is where the diagonal line comes from
    cov = torch.where(
        # Where the diagonal is negative
        torch.diag_embed(torch.diagonal(cov, dim1=-1, dim2=-2)) < 0,
        torch.abs(cov),
        cov,
    )
    try:
        cholesky = torch.cholesky(cov.float())
    except RuntimeError:
        print("Singular CUDA")
        cholesky = torch.cholesky(cov.float() + torch.eye(2))
    # NOTE: removed loc_scale
    cholesky = cholesky
    if torch.isnan(cholesky).any():
        print("Not Symmetric Positive Semi-definite")
        __import__("pdb").set_trace()

        # @rayandrews
        # this is inplace operation
        # will result error in backprop!
        # cholesky[torch.isnan(cholesky)] = 0

        # below is the safe version
        cholesky = torch.where(
            torch.isnan(cholesky), torch.zeros_like(cholesky), cholesky
        )

    return mean, cholesky


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
