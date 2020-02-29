import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.interpolate import InterpolateModule as Interpolate
from utils.gaussian import flipDiagonal, left_pad

# from data.config import cfg


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

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
        gridShape = list(grid.Shape)

        locShape = list(loc.shape)
        originalShape = list(original.shape)

        # Dim: Batch, Priors, 3, img_h, img_w
        original = original.expand([locShape[1]] + originalShape).permute(1, 0, 2, 3, 4)
        # Dim: Batch*Priors, 3, img_h, img_w
        # unsup: changed view to reshape
        original = original.reshape(-1, *originalShape[1:])

        # grid= grid.reshape(-1, *originalShape[1:])

        # batch*priors, img_h,img_w,2
        grid = grid.reshape(-1, *gridShape[2:])
        print(original.shape)
        print(grid.shape)

        # Dim: Batch*priors,3,H,W
        sampled = F.grid_sample(original, grid)

        result = self.encoder(sampled)
        result = self.decoder(result)
        result = F.interpolate(result, size=cfg.sampling_grid, mode="bilinear")

        # Dim Batch*Prior, 3, H, W
        loss = F.mse_loss(sampled, result, reduction="none")

        # Dim Batch*Prior
        loss = loss.sum(dim=(1, 2, 3))

        # Conf: batch,prior (only 1 class)
        # Confidence in foreground
        # conf = predictions["conf"][:, :, 1]

        # Batch*prior
        # conf = conf.view(-1)

        # Scale loss by confidence, and by the number of Batches and Priors
        return loss
        # return loss / (locShape[0] * locShape[1])


def samplingGrid(loc):
    # img_dim in tensor form
    print(loc.shape)

    # Batch, Priors
    locShape = list(loc.shape)[:-1]

    # H, W
    gridShape = cfg.sampling_grid

    # Batch, Priors, 2
    mean = loc[:, :, :2]
    # ((Batch,Priors,2),(Batch,Priors,2))
    cov = (loc[:, :, 2:4], left_pad(loc[:, :, 4:]))
    # (Batch,Priors, 2(add this new dimension),2)
    cov = flipDiagonal(torch.stack(cov, dim=2))

    # Ensure positive semi-definite
    cov = cov + (cfg.positive * torch.eye(2))

    # Batch*Priors, 2,2
    cov = cov.view(-1, 2, 2)

    # Batch*Priors, 2,2
    cholesky = torch.cholesky(cov)

    # Batch*Priors, 2
    mean = mean.view(-1, 2)

    j, i = torch.meshgrid(
        torch.linspace(-1, 1, cfg.sampling_grid[0]),
        torch.linspace(-1, 1, cfg.sampling_grid[1]),
    )
    i, j = i.contiguous(), j.contiguous()

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
    reshaped_coords = transformed_coords.view([-1] + cfg.sampling_grid + [2])
    return reshaped_coords
