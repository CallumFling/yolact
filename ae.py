import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.interpolate import InterpolateModule as Interpolate
from utils.gaussian import gauss_loc

from data import cfg
import pdb


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        modules = []
        in_channels = cfg.ae_dim
        dims = [cfg.fpn.num_features] + cfg.ae_dim

        # adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        for i in range(len(dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=dims[i],
                        out_channels=dims[i + 1],
                        **cfg.encoder_layer_params
                    ),
                    nn.ReLU(),
                )
            )

        self.encoder = nn.Sequential(*modules)

        modules = []
        # 3 for RGB
        dims = list(reversed(cfg.ae_dim)) + [3]
        args = {
            "mode": "bilinear",
            "scale_factor": cfg.decoder_interpolate_scale,
            "recompute_scale_factor": False,
            "align_corners": False,
        }

        # Still need to interpolate, because size too small
        for i in range(len(dims) - 1):
            modules.append(
                nn.Sequential(
                    Interpolate(**args),
                    nn.ConvTranspose2d(
                        in_channels=dims[i],
                        out_channels=dims[i + 1],
                        **cfg.decoder_layer_params
                    ),
                    # if Last, use Sigmoid
                    (nn.ReLU() if i != (len(dims) - 2) else nn.Sigmoid()),
                )
            )
        self.decoder = nn.Sequential(*modules)

    def forward(self, original, proto_x, loc):
        # The input should be of size [batch_size, 3, img_h, img_w]
        # conf shape: torch.size(batch_size,num_priors,num_classes)
        # loc shape: torch.size(batch_size,num_priors,5)

        # batch, priors, img_h, img_w, 2
        feature_grid = samplingGrid(loc, cfg.feature_sampling_grid)
        feature_grid_shape = list(feature_grid.shape)
        original_grid = samplingGrid(loc, cfg.original_sampling_grid)
        original_grid_shape = list(original_grid.shape)

        # locShape = list(loc.shape)
        proto_x_shape = list(proto_x.shape)
        original_shape = list(original.shape)

        # Dim: Batch, Priors, 3, img_h, img_w
        proto_x = proto_x.expand([feature_grid_shape[1]] + proto_x_shape).permute(
            1, 0, 2, 3, 4
        )
        original = original.expand([original_grid_shape[1]] + original_shape).permute(
            1, 0, 2, 3, 4
        )
        # Dim: Batch*Priors, 3, img_h, img_w
        # unsup: changed view to reshape
        proto_x = proto_x.reshape(-1, *proto_x_shape[1:])
        original = original.reshape(-1, *original_shape[1:])

        # batch*priors, img_h,img_w,2
        feature_grid = feature_grid.reshape(-1, *feature_grid_shape[2:])
        original_grid = original_grid.reshape(-1, *original_grid_shape[2:])

        # @rayandrew
        # so my analysis here
        # F.grid_sample with default mode=bilinear is exploded
        # no oom issue here

        # NOTE: Grid_Sample behavior
        # [[1,2]
        # [3,4]]

        # [-1,-1] -> 1
        # [1,-1] -> 2
        # [-1,1] -> 2
        # [1,1] ->4

        # NOTE: Using padding border, not bilinear
        # Dim: Batch*priors,3,H,W
        feature_sample = F.grid_sample(
            proto_x,
            feature_grid,
            align_corners=False,
            padding_mode="border",
            mode="bilinear",
        )  # this seem the fixes
        original_sample = F.grid_sample(
            original,
            original_grid,
            align_corners=False,
            padding_mode="border",
            mode="bilinear",
        )  # this seem the fixes

        result = self.encoder(feature_sample)
        result = self.decoder(result)

        # NOTE: Using Bilinear
        result = F.interpolate(
            result,
            size=cfg.original_sampling_grid,
            mode="bilinear",
            align_corners=False,
        )

        # Dim Batch*Prior, 3, H, W
        # NOTE: fixed flipped input target
        loss = F.mse_loss(result, original_sample, reduction="none")

        # Dim Batch*Prior
        loss = torch.mean(loss, dim=(1, 2, 3))
        # loss = loss.sum(dim=(1, 2, 3))

        # Dim Batch,Priors
        loss = loss.view(*feature_grid_shape[:2])

        # Scale loss by confidence, and by the number of Batches and Priors
        if cfg.use_amp:
            return loss.half()
        return loss
        # return loss / (locShape[0] * locShape[1])


def samplingGrid(loc, gridShape):
    # img_dim in tensor form
    # Batch, Priors
    locShape = list(loc.shape)[:-1]

    # H, W
    # gridShape = cfg.sampling_grid

    # mean, cov = gauss_loc(loc)
    mean, cholesky = gauss_loc(loc)
    # Batch*Priors, 2,2
    # cov = cov.view(-1, 2, 2)
    cholesky = cholesky.view(-1, 2, 2)

    # Batch*Priors, 2
    mean = mean.view(-1, 2)

    j, i = torch.meshgrid(
        torch.linspace(-1, 1, gridShape[0]), torch.linspace(-1, 1, gridShape[1]),
    )
    i, j = i.contiguous(), j.contiguous()
    # if cfg.use_amp:
    # i, j = i.half(), j.half()

    # mesh,2
    # becomes tensor of [(0,0),(1,0),(2,0)...]
    coordinate_list = torch.stack((i.view(-1), j.view(-1)), dim=1)

    # Coordinates, Batch*Priors, 2
    # dab to be able to add the mean
    # Batch, 2,2 vs mesh, 2 -> Mesh, Batch, 2
    transformed_coords = torch.einsum("abc,dc->dab", cholesky, coordinate_list) + mean

    # Turn into ADB: Batch*Prior, Mesh, 2
    transformed_coords = transformed_coords.permute(1, 0, 2)

    # Dim Batch*Prior, H,W,2->batch, priors, img_h, img_w, 2
    reshaped_coords = transformed_coords.view([locShape[0]] + [-1] + gridShape + [2])
    return reshaped_coords
