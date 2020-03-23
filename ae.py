import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.interpolate import InterpolateModule as Interpolate
from utils.gaussian import gauss_loc, sampling_grid

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
        feature_grid = sampling_grid(loc, cfg.feature_sampling_grid)
        feature_grid_shape = list(feature_grid.shape)
        # batch*priors, img_h,img_w,2
        feature_grid = feature_grid.reshape(-1, *feature_grid_shape[2:])

        # locShape = list(loc.shape)
        proto_x_shape = list(proto_x.shape)

        # Dim: Batch, Priors, 3, img_h, img_w
        proto_x = proto_x.expand([feature_grid_shape[1]] + proto_x_shape).permute(
            1, 0, 2, 3, 4
        )
        # Dim: Batch*Priors, 3, img_h, img_w
        # unsup: changed view to reshape
        proto_x = proto_x.reshape(-1, *proto_x_shape[1:])

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

        result = self.encoder(feature_sample)
        result = self.decoder(result)

        reconstruction_grid = sampling_grid(loc, cfg.background_shape, inverse=True)
        reconstruction_grid_shape = list(reconstruction_grid.shape)
        # batch*priors, img_h,img_w,2
        reconstruction_grid = reconstruction_grid.reshape(
            -1, *reconstruction_grid_shape[2:]
        )

        reconstruction = F.grid_sample(
            result,
            reconstruction_grid,
            align_corners=False,
            padding_mode="zeros",
            mode="bilinear",
        )

        # Batch, Priors, 3, i,j
        reconstruction = reconstruction.view(
            *reconstruction_grid_shape[:2], 3, *reconstruction_grid_shape[2:4]
        )

        __import__("pdb").set_trace()
        return reconstruction
        # return loss / (locShape[0] * locShape[1])


