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

        self.encoder_element = [
            # self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=6, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, kernel_size=6, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 4, kernel_size=6, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 2, kernel_size=6, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # )
        ]

        def encoder(x):
            for i, v in enumerate(self.encoder_element):
                print(i)
                x = v(x)
            return x

        self.encoder = encoder

        self.decoder = nn.Sequential(
            Interpolate(mode="bilinear", scale_factor=2),
            nn.ConvTranspose2d(2, 4, kernel_size=3),
            nn.ReLU(True),
            Interpolate(mode="bilinear", scale_factor=2),
            nn.ConvTranspose2d(4, 8, kernel_size=3),
            nn.ReLU(True),
            Interpolate(mode="bilinear", scale_factor=2),
            nn.ConvTranspose2d(8, 16, kernel_size=3),
            nn.ReLU(True),
            Interpolate(mode="bilinear", scale_factor=2),
            nn.ConvTranspose2d(16, 32, kernel_size=3),
            nn.ReLU(True),
            Interpolate(mode="bilinear", scale_factor=2),
            nn.ConvTranspose2d(32, 3, kernel_size=3),
            # nn.ReLU(True),
            nn.Sigmoid(),
        )

    def forward(self, original, loc):
        # The input should be of size [batch_size, 3, img_h, img_w]
        # conf shape: torch.size(batch_size,num_priors,num_classes)
        # loc shape: torch.size(batch_size,num_priors,5)

        # batch, priors, img_h, img_w, 2
        grid = samplingGrid(loc)
        gridShape = list(grid.shape)

        # locShape = list(loc.shape)
        originalShape = list(original.shape)

        # Dim: Batch, Priors, 3, img_h, img_w
        original = original.expand([gridShape[1]] + originalShape).permute(
            1, 0, 2, 3, 4
        )
        # Dim: Batch*Priors, 3, img_h, img_w
        # unsup: changed view to reshape
        original = original.reshape(-1, *originalShape[1:])

        # grid= grid.reshape(-1, *originalShape[1:])

        # batch*priors, img_h,img_w,2
        grid = grid.reshape(-1, *gridShape[2:])

        print(
            "IS NAN, original {} grid {}".format(
                torch.isnan(original).any(), torch.isnan(grid).any()
            )
        )
        print(
            "IS INF, original {} grid {}".format(
                torch.isinf(original).any(), torch.isinf(grid).any()
            )
        )

        # @rayandrew
        # so my analysis here
        # F.grid_sample with default mode=bilinear is exploded
        # no oom issue here

        # NOTE: Using padding border, not bilinear
        # Dim: Batch*priors,3,H,W
        sampled = F.grid_sample(
            original, grid, align_corners=False, padding_mode="border", mode="bilinear"
        )  # this seem the fixes

        print("Sampled NAN", torch.isnan(sampled).any())

        # sampled[torch.isinf(sampled)] = 0
        # safe_sampled = torch.where(torch.isfinite(sampled), torch.zeros_like(sampled), sampled)

        # try:
        print(
            "Original {} Grid {} Sampled shape {}".format(
                original.shape, grid.shape, sampled.shape
            )
        )
        # print("SAMPLED {}".format(sampled))
        result = self.encoder(sampled)
        # except RuntimeError:
        # pdb.set_trace()
        result = self.decoder(result)
        # NOTE: Using Nearest, not Bilinear
        result = F.interpolate(
            result, size=cfg.sampling_grid, mode="bilinear", align_corners=False
        )

        # Dim Batch*Prior, 3, H, W
        loss = F.mse_loss(sampled, result, reduction="none")

        # Dim Batch*Prior
        loss = torch.mean(loss, dim=(1, 2, 3))
        # loss = loss.sum(loss)

        # Conf: batch,prior (only 1 class)
        # Confidence in foreground
        # conf = predictions["conf"][:, :, 1]

        # Batch*prior
        # conf = conf.view(-1)

        # Scale loss by confidence, and by the number of Batches and Priors
        if cfg.use_amp:
            return loss.half()
        return loss
        # return loss / (locShape[0] * locShape[1])


def samplingGrid(loc):
    # img_dim in tensor form
    # Batch, Priors
    locShape = list(loc.shape)[:-1]

    # H, W
    gridShape = cfg.sampling_grid

    # mean, cov = gauss_loc(loc)
    mean, cholesky = gauss_loc(loc)
    # Batch*Priors, 2,2
    # cov = cov.view(-1, 2, 2)
    cholesky = cholesky.view(-1, 2, 2)

    # Batch*Priors, 2,2
    # if cfg.use_amp:
    # cholesky = torch.cholesky(cov.float()).half()
    # else:

    # Batch*Priors, 2
    mean = mean.view(-1, 2)

    j, i = torch.meshgrid(
        torch.linspace(-1, 1, cfg.sampling_grid[0]),
        torch.linspace(-1, 1, cfg.sampling_grid[1]),
    )
    i, j = i.contiguous(), j.contiguous()
    # if cfg.use_amp:
    # i, j = i.half(), j.half()

    # mesh,2
    # becomes tensor of [(0,0),(1,0),(2,0)...]
    coordinate_list = torch.stack((i.view(-1), j.view(-1)), dim=1)

    # Dim: batch*priors, coordinate (i,j)
    # transformed_coords = torch.einsum("abc,ac->ab", cholesky, coordinate_list) + mean

    # Coordinates, Batch*Priors, 2
    # dab to be able to add the mean

    # Batch, 2,2 vs mesh, 2 -> Mesh, Batch, 2
    transformed_coords = torch.einsum("abc,dc->dab", cholesky, coordinate_list) + mean

    # Turn into ADB: Batch*Prior, Mesh, 2
    transformed_coords = transformed_coords.permute(1, 0, 2)

    # Dim Batch*Prior, H,W,2->batch, priors, img_h, img_w, 2
    reshaped_coords = transformed_coords.view(
        [locShape[0]] + [-1] + cfg.sampling_grid + [2]
    )
    return reshaped_coords
