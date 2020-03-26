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
    sin, cos = (
        torch.sin(theta / cfg.gauss_sensitivity),
        torch.cos(theta / cfg.gauss_sensitivity),
    )

    # shape: torch.shape(batch,num_priors,2)
    rot_0 = torch.stack((cos, -sin), dim=2)
    rot_1 = torch.stack((sin, cos), dim=2)

    # shape: torch.shape(batch,num_priors,2,2)
    rot = torch.stack((rot_0, rot_1), dim=2)
    return rot


# @snoop
def gauss_loc(loc, priors, inverse=False):
    # NOTE: Dumb Trick: Periodic Function with Modulo, and with defined
    # NOTE: this is calculated twice, cache to optimize
    # NOTE: the name Cholesky is wrong, since the variable "cholesky" is not triangular

    # Σ is equal to RSS(R-1)
    # Cholesky=T=RS
    # calculate this instead
    # loc shape: torch.size(batch_size,num_priors,5)
    # locShape = list(loc.shape)
    # priorsShape = list(batch,priors, 4)
    priors_mean = priors[:, :, :2]
    mean = loc[:, :, :2]
    # mean = torch.tanh(mean)
    mean = (
        torch.sin(mean / cfg.gauss_sensitivity + torch.asin(priors_mean))
        * cfg.gauss_wrap
    )

    theta = loc[:, :, 2]
    rot = rotation_matrix(theta)
    if inverse:
        # Since inverse of rotation_matrix is transpose
        rot = rot.permute(0, 1, 3, 2)

    scale_coeff = loc[:, :, 3:5]
    scale_coeff = (
        torch.sin(scale_coeff / cfg.gauss_sensitivity) * cfg.max_scale
        + cfg.max_scale
        + cfg.positive
    )

    if inverse:
        scale_coeff = 1 / scale_coeff
    scale = torch.diag_embed(scale_coeff)

    if not inverse:
        rs = rot @ scale
    else:
        sr = scale @ rot

    # mean shape: torch.size(batch_size,num_priors,2)
    # cholesky shape: torch.size(batch_size,num_priors,2,2)
    if not inverse:
        return mean, rs
    return mean, sr


# White as in white noise
def white_coordinates(shape):
    j, i = torch.meshgrid(
        torch.linspace(-1, 1, shape[0]), torch.linspace(-1, 1, shape[1]),
    )
    i, j = i.contiguous(), j.contiguous()
    # mesh,2
    # becomes tensor of [(0,0),(1,0),(2,0)...]
    coordinate_list = torch.stack((i.view(-1), j.view(-1)), dim=1)
    return coordinate_list


# @snoop
def mahalanobisGrid(maskShape, loc, priors):
    locShape = list(loc.shape)[:-1]
    # rs as in Rotate Scale
    mean, rs = gauss_loc(loc, priors)

    # Ensuring Positive-Definite https://stackoverflow.com/a/40575354/10702372
    # Batch, priors, 2,2
    covariance = rs @ rs.permute(0, 1, 3, 2) + cfg.positive * torch.eye(2)

    # Ensuring non-singular matrix
    # batch, priors
    # singular = covariance.det() == 0
    # covariance[singular] = torch.eye(2) * cfg.positive

    cholesky = torch.cholesky(covariance)
    # print("Is symmetric:", torch.all(covariance.permute(0, 1, 3, 2) == covariance))
    # eig = torch.symeig(covariance)[0]
    # print("Is positive definite:", torch.all(eig > 0))

    # Resize to Batch Muahalanobis Specs
    cholesky = cholesky.view(-1, 1, 2, 2)
    mean = mean.view(-1, 1, 2)

    coordinate_list = white_coordinates(maskShape)

    # NOTE: ORDER OF OPERATIONS HERE MIGHT BE WRONG
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


def unnormalGaussian(maskShape, loc, priors):
    result = torch.pow(
        torch.tensor([math.e]),
        mahalanobisGrid(maskShape=maskShape, loc=loc, priors=priors),
    )
    if cfg.use_amp:
        return result.half()
    return result


def sampling_grid(loc, gridShape, priors, inverse=False):
    # img_dim in tensor form
    # Batch, Priors
    locShape = list(loc.shape)[:-1]

    # H, W
    # gridShape = cfg.sampling_grid

    # mean, cov = gauss_loc(loc)
    mean, rs = gauss_loc(loc, priors, inverse=inverse)
    # Batch*Priors, 2,2
    # cov = cov.view(-1, 2, 2)
    rs = rs.view(-1, 2, 2)

    # Batch*Priors, 2
    mean = mean.view(-1, 2)

    coordinate_list = white_coordinates(gridShape)

    # Coordinates, Batch*Priors, 2
    # dab to be able to add the mean
    # Batch, 2,2 vs mesh, 2 -> Mesh, Batch, 2

    if not inverse:
        transformed_coords = torch.einsum("abc,dc->dab", rs, coordinate_list) + mean
    else:
        # coordinate_list shape: mesh,2 -> mesh, batch*priors, 2
        coordinate_list = coordinate_list.unsqueeze(1).repeat(
            1, locShape[0] * locShape[1], 1
        )
        transformed_coords = torch.einsum(
            # RS shape: batch*priors, 2,2
            # mean shape: batch*priors, 2
            "abc,dac->dab",
            rs,
            coordinate_list - mean,
        )

    # Turn into ADB: Batch*Prior, Mesh, 2
    transformed_coords = transformed_coords.permute(1, 0, 2)

    # Dim Batch*Prior, H,W,2->batch, priors, img_h, img_w, 2
    reshaped_coords = transformed_coords.view(locShape + gridShape + [2])
    return reshaped_coords
