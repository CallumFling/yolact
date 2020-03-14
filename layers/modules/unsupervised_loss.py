# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from data.config import cfg
from utils import gaussian, timer
import math
from ae import AutoEncoder
import pdb
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter


def unsup_writer(Writer):
    global writer
    writer = Writer


def unsup_iter(Iteration):
    global iteration
    iteration = Iteration


# https://stackoverflow.com/questions/55628014/indexing-a-3d-tensor-using-a-2d-tensor
def twoIndexThree(tensor, index):
    return tensor[torch.arange(tensor.shape[0]).unsqueeze(-1), index]


class UnsupervisedLoss(nn.Module):
    def __init__(self):
        super(UnsupervisedLoss, self).__init__()
        self.variance = VarianceLoss()
        self.autoencoder = AutoEncoder()
        self.num_classes = cfg.num_classes  # Background included
        self.background_label = 0
        self.top_k = cfg.nms_top_k
        self.nms_thresh = cfg.nms_thresh
        self.conf_thresh = cfg.nms_conf_thresh

        if self.nms_thresh <= 0:
            raise ValueError("nms_threshold must be non negative.")

    def forward(self, original, predictions):
        # loc shape: torch.size(batch_size,num_priors,6)
        loc_data = predictions["loc"]
        # conf shape: torch.size(batch_size,num_priors,num_classes)
        conf_data = predictions["conf"]
        # Softmaxed confidence in Foreground
        # Shape: batch,num_priors
        conf_data = F.softmax(conf_data, dim=2)[:, :, 1]
        # masks shape: torch.size(batch_size,num_priors,mask_dim)
        mask_data = predictions["mask"]
        # proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)
        proto_data = predictions["proto"]

        losses = {}
        with timer.env("Detect"):
            all_results = self.detect(conf_data, loc_data, mask_data)
            keep = all_results["keep"]
            # AE Scaled loss needs non-kept Detections
            # NOTE: Loss scaling to remove Backward errors
            losses["ae_loss"] = self.ae_scaled_loss(
                original,
                all_results["iou"],
                all_results["keep"],
                all_results["loc"],
                all_results["conf"],
            )

            # IoU not included because not needed by variance
            out = {
                "loc": twoIndexThree(all_results["loc"], keep),
                "mask": twoIndexThree(all_results["mask"], keep),
                "conf": torch.gather(all_results["conf"], 1, keep),
                "proto": proto_data,
            }

            losses["variance_loss"] = self.variance(
                original, out["loc"], out["mask"], out["conf"], out["proto"]
            )
            out["losses"] = losses

        return out

    def detect(self, conf, loc, mask):
        # IoU Threshold, Conf_Threshold, IoU_Thresh
        """
        NO BATCH
        boxes=loc Shape: [num_priors, 4]
        masks: [batch, num_priors, mask_dim]
        """
        # unsup: Background isn't NMSed
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        top_k_conf = cfg.nms_top_k_conf
        top_k_iou = cfg.nms_top_k_iou

        # __import__("pdb").set_trace()
        # conf shape: torch.size(batch_size,num_priors)
        sorted_conf, sorted_conf_idx = torch.topk(
            conf, top_k_conf, dim=-1, largest=True
        )
        # loc with size [batch,num_priors, 4], or 5 for unsup
        sorted_loc = twoIndexThree(loc, sorted_conf_idx)
        # masks shape: Shape: [num_priors, mask_dim]
        sorted_mask = twoIndexThree(mask, sorted_conf_idx)

        # Dim: Batch, Detections, i,j
        gauss = gaussian.unnormalGaussian(maskShape=cfg.iou_gauss_dim, loc=sorted_loc)

        gaussShape = list(gauss.shape)
        # jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        gauss_rows = gauss.view(
            gaussShape[0], 1, gaussShape[1], *gaussShape[2:]
        ).repeat(1, gaussShape[1], 1, 1, 1)
        gauss_cols = gauss.view(
            gaussShape[0], gaussShape[1], 1, *gaussShape[2:]
        ).repeat(1, 1, gaussShape[1], 1, 1)
        # Batch, Detections, Detections, 2, I,J
        gauss_grid = torch.stack([gauss_rows, gauss_cols], dim=3)

        # [0] is to remove index
        gauss_intersection = torch.sum(torch.min(gauss_grid, dim=3)[0], dim=[3, 4])
        gauss_union = torch.sum(torch.max(gauss_grid, dim=3)[0], dim=[3, 4])

        # Batch, Detections, Detections
        gauss_iou = gauss_intersection / (gauss_union)

        # Batch, Detections
        iou_max, _ = gauss_iou.triu(diagonal=1).max(dim=1)

        # From lowest to highest IoU
        _, sorted_iou_idx = torch.topk(iou_max, top_k_iou, dim=-1, largest=False)

        return {
            "iou": gauss_iou,
            "loc": sorted_loc,
            "mask": sorted_mask,
            "conf": sorted_conf,
            "keep": sorted_iou_idx,
        }

    def ae_scaled_loss(self, original, iou, keep, loc, conf):
        # AE input should be of size [batch_size, 3, img_h, img_w]
        # conf shape: torch.size(batch_size,num_priors)

        # conf matrix: torch.size(batch_size,num_priors,num_priors)
        conf_matrix = conf.unsqueeze(1).permute(0, 2, 1) @ conf.unsqueeze(1)
        # Remove Lower Triangle and Diagonal
        final_scale = iou * conf_matrix.triu(1)

        # Dim batch,priors
        # try:
        ae_loss = self.autoencoder(original, twoIndexThree(loc, keep))
        # except RuntimeError:
        # pdb.set_trace()
        # ae_loss but in scores
        # batch, priors, priors
        # pdb.set_trace()
        ae_grid = torch.zeros_like(final_scale)
        ae_grid[torch.arange(ae_grid.shape[0]).unsqueeze(-1), keep] = ae_loss.unsqueeze(
            2
        ).repeat(1, 1, ae_grid.size(2))
        ae_grid = ae_grid * final_scale

        # Divide by Priors
        return torch.mean(ae_grid)


class MyException(Exception):
    pass


class VarianceLoss(nn.Module):
    def __init__(self):
        super(VarianceLoss, self).__init__()
        pass

    def forward(self, original, loc, mask, conf, proto):
        log = iteration % 1 == 0
        # original is [batch_size, 3, img_h, img_w]
        original = original.float()

        resizeShape = list(original.shape)[-2:]

        # proto* shape: torch.size(batch,mask_h,mask_w,mask_dim)
        # proto_shape = list(predictions[0]["proto"].shape)[:2]
        proto_shape = list(proto.shape)[1:3]

        unnormalGaussian = gaussian.unnormalGaussian(maskShape=proto_shape, loc=loc)

        # Display multiple images in Batch[0]
        if log:
            writer.add_images(
                "variance/0_unnormalGaussian",
                unnormalGaussian[0].unsqueeze(-1),
                iteration,
                dataformats="NHWC",
            )

        # Dim: Batch, Anchors, i, j
        assembledMask = gaussian.lincomb(proto=proto, masks=mask)
        if log:
            writer.add_images(
                "variance/1_lincomb",
                assembledMask[0].unsqueeze(-1),
                iteration,
                dataformats="NHWC",
            )
        assembledMask = torch.sigmoid(assembledMask)

        attention = assembledMask * unnormalGaussian
        if log:
            writer.add_images(
                "variance/2_attention",
                attention[0].unsqueeze(-1),
                iteration,
                dataformats="NHWC",
            )

        # conf shape: torch.size(batch_size,num_priors)
        # Batch, Anchors, i,j
        maskConfidence = torch.einsum("abcd,ab->abcd", attention, conf)
        if log:
            writer.add_images(
                "variance/3_maskConfidence",
                maskConfidence[0].unsqueeze(-1),
                iteration,
                dataformats="NHWC",
            )

        # unsup: REMOVE CHANNEL DIMENSION HERE, since Pad_sequence is summed, it does nothing
        # Confidence in background, see desmos
        # Dim: batch, h, w
        finalConf = 1 - torch.sum(
            (maskConfidence ** 2)
            / (
                torch.sum(maskConfidence, dim=1, keepdim=True).repeat(
                    1, maskConfidence.size(1), 1, 1
                )
                + cfg.positive
            ),
            dim=1,
        )

        finalConf = torch.where(
            torch.isnan(finalConf), torch.zeros_like(finalConf), finalConf
        )
        if log:
            writer.add_images(
                "variance/4_finalConf",
                finalConf.unsqueeze(1),
                iteration,
                dataformats="NCHW",
            )

        # unsup: Interpolation may be incorrect
        # Dim batch, img_h, img_w
        resizedConf = F.interpolate(
            finalConf.unsqueeze(1), resizeShape, mode="bilinear", align_corners=False
        )[:, 0]

        if log:
            writer.add_images(
                "variance/5_resizedConf",
                resizedConf.unsqueeze(1),
                iteration,
                dataformats="NCHW",
            )
        # if cfg.use_amp:
        # finalConf = finalConf.half()
        # resizedConf = resizedConf.half()

        # unsup: AGGREGATING RESULTS BETWEEN BATCHES HERE
        # Dim: h, w
        totalConf = torch.sum(resizedConf, dim=0)
        if log:
            writer.add_image(
                "variance/6_totalConf", totalConf, iteration, dataformats="HW"
            )

        # NOTE MAYBE WEIGHTEd MEAN NOT NORMALIZed OOPH RITE
        # Dim 3, img_h, img_w
        weightedMean = torch.einsum("abcd,acd->bcd", original, resizedConf) / totalConf
        if log:
            writer.add_image("variance/7_weightedMean", weightedMean, iteration)

        # Dim: batch, 3, img_h, img_w
        squaredDiff = (original - weightedMean) ** 2
        # if cfg.use_amp:
        # squaredDiff = squaredDiff.half()

        # Batch,3,img_h, image w
        weightedDiff = torch.einsum("abcd,acd->abcd", squaredDiff, resizedConf)
        if log:
            writer.add_images("variance/8_weightedDiff", weightedDiff, iteration)

        weightedVariance = weightedDiff / (totalConf + cfg.positive)
        if log:
            writer.add_images(
                "variance/9_weightedVariance", weightedVariance, iteration
            )

        # NOTE: Arbitrary Loss Scaling
        result = torch.sum(weightedVariance) / (proto.shape[1]) * original.shape[0]

        # if cfg.use_amp:
        # return result.half()

        # Normalize by number of elements in original image
        return result
