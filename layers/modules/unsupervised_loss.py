# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from data.config import cfg
from utils import gaussian, timer
import math
from ae import AutoEncoder


class UnsupervisedLoss(nn.Module):
    def __init__(self):
        super(UnsupervisedLoss, self).__init__()
        self.variance = VarianceLoss()
        self.autoencoder = AutoEncoder()
        self.iou_layer = torch.nn.Sequential(
            torch.nn.Linear(10, cfg.iou_middle_features),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.iou_middle_features, 1),
            torch.nn.Sigmoid(),
        )
        self.num_classes = cfg.num_classes  # Background included
        self.background_label = 0
        self.top_k = cfg.nms_top_k
        self.nms_thresh = cfg.nms_thresh
        self.conf_thresh = cfg.nms_conf_thresh

        if self.nms_thresh <= 0:
            raise ValueError("nms_threshold must be non negative.")

    def forward(self, original, predictions):
        # loc shape: torch.size(batch_size,num_priors,4)
        loc_data = predictions["loc"]
        # conf shape: torch.size(batch_size,num_priors,num_classes)
        conf_data = predictions["conf"]
        # masks shape: torch.size(batch_size,num_priors,mask_dim)
        mask_data = predictions["mask"]
        # proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)
        proto_data = predictions["proto"]

        losses = {"iou_loss": 0, "ae_loss": 0}
        out = []

        with timer.env("Detect"):
            batch_size = loc_data.size(0)

            # conf_preds shape: torch.size(batch_size,num_classes,num_priors)
            conf = (
                conf_data.view(batch_size, -1, self.num_classes)
                .transpose(2, 1)
                .contiguous()
            )

            for batch_idx in range(batch_size):
                # decoded boxes with size [num_priors, 4]
                all_results = self.detect(
                    conf, loc_data[batch_idx], mask_data[batch_idx]
                )
                if result is not None and proto_data is not None:
                    result["proto"] = proto_data[batch_idx]
                else:
                    raise ValueError("No proto or no Result")
                losses["iou_loss"] += result["iou_loss"]
                # AE Scaled loss needs non-kept Detections
                losses["ae_loss"] += self.ae_scaled_loss(
                    original[batch_idx],
                    all_results["iou"],
                    all_results["keep"],
                    all_results["loc"],
                    all_results["conf"],
                )
                keep = all_results["keep"]

                # IoU not included because not needed by variance
                filtered_result = {
                    "loc": loc[keep],
                    "mask": mask[keep],
                    "conf": conf[keep],
                }
                out.append(result)

        # losses["ae_loss"] = predictions["ae_loss"]
        losses["variance_loss"] = self.variance(original, out)
        return losses

    def detect(self, conf, loc, mask):
        """
        NO BATCH
        boxes=loc Shape: [num_priors, 4]
        masks: [batch, num_priors, mask_dim]
        """
        # unsup: Background isn't NMSed
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        # conf shape: torch.size(1,num_priors)
        if conf.size(0) > 1:
            print(cur_scores.shape)
            raise MyException("WHY MULTIPLE CLASSES")

        # cur_scores shape: torch.size(num_priors)
        conf = conf[1, :]

        # filter below confidence
        keep = conf_scores > self.conf_thresh

        # filtered num_priors
        filtered_conf = conf[keep]
        # decoded boxes with size [num_priors, 4], or 5 for unsup
        filtered_loc = loc[keep, :]
        # shape torch.size(num_priors,mask_dim)
        filtered_masks = mask_data[keep, :]

        if filtered_conf.size(1) == 0:
            return None

        iou_grid, keep, loc, mask, conf, iou_loss = self.unsup_nms(
            filtered_conf,
            filtered_loc,
            filtered_masks,
            self.nms_thresh,
            self.top_k,
            train_iou=True,
        )

        return {
            "iou": iou,
            "loc": loc,
            "mask": mask,
            "conf": conf,
            "keep": keep,
            "iou_loss": iou_loss,
        }

    def unsup_nms(
        self,
        conf,
        loc,
        mask,
        iou_threshold: float = 0.5,
        top_k: int = 200,
        second_threshold: bool = False,
        train_iou=False,
    ):
        # Performs sorting by confidence, returns conf,loc,mask of these sorted calculating IoU grid, index of passing NMS is returned

        # cur_scores shape: torch.size(num_priors)
        # sort by confidence
        conf, idx = conf.sort(descending=True)

        # get top_k scores
        idx = idx[:top_k].contiguous()
        conf = conf[:top_k]
        # loc with size [num_priors, 4], or 5 for unsup
        loc = loc[idx, :]
        # masks shape: Shape: [num_priors, mask_dim]
        mask = mask[idx, :]

        # jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        dup_rows = loc.view(1, loc.size(0), loc.size(1)).repeat(loc.size(0), 1, 1)
        dup_cols = loc.view(loc.size(0), 1, loc.size(1)).repeat(1, loc.size(0), 1)
        # num_dets, num_dets, 10
        grid = torch.cat((dup_rows, dup_cols), dim=-1)

        self.iou_layer.requires_grad = False

        # num_dets,num_dets,1
        iou_grid = self.iou_layer(grid)

        iou_max, _ = iou.triu(diagonal=1).max(dim=0)

        # Now just filter out the ones higher than the threshold
        # Tensor of shape, num_dets
        keep = iou_max <= iou_threshold
        # boxes = boxes[keep]
        # masks = masks[keep]
        # scores = scores[keep]

        iou_loss = 0
        if train_iou:
            self.iou_layer.requires_grad = True
            # Dim: Detections, i,j
            gauss = gaussian.unnormalGaussian(
                maskShape=cfg.iou_layer_train_dim,
                loc=boxes[keep, :][: cfg.gauss_iou_samples].unsqueeze(0),
            )[0]

            # Priors, i,j
            gaussShape = list(gauss.shape)

            # jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
            gauss_rows = gauss.view(1, gaussShape[0], *gaussShape[1:]).repeat(
                gaussShape[0], 1, 1, 1
            )
            gauss_cols = boxes.view(gaussShape[0], 1, *gaussShape[1:]).repeat(
                1, gaussShape[0], 1, 1
            )
            # Detections, Detections, 2, I,J
            gauss_grid = torch.stack([gauss_rows, gauss_cols], dim=2)

            # [0] is to remove index
            gauss_intersection = torch.sum(torch.min(gauss_grid, dim=2)[0], dim=[2, 3])
            gauss_union = torch.sum(torch.max(gauss_grid, dim=2)[0], dim=[2, 3])
            gauss_iou = gauss_intersection / gauss_union

            # if error here, may be because of unsqueeze                       [0]
            iou_loss = F.mse_loss(
                iou[keep, keep][: cfg.gauss_iou_samples, : cfg.gauss_iou_samples],
                gauss_iou,
            )

        # Omit this because of iou_grid
        # Only keep the top cfg.max_num_detections highest scores across all classes
        # scores, idx = scores.sort(0, descending=True)
        # idx = idx[: cfg.max_num_detections]
        # scores = scores[: cfg.max_num_detections]

        # boxes = boxes[idx]
        # masks = masks[idx]

        return iou_grid, keep, loc, mask, conf, iou_loss

    def ae_scaled_loss(self, original, iou, keep, loc, conf):
        # AE input should be of size [batch_size, 3, img_h, img_w]
        conf_matrix = scores.unsqueeze(0).t() @ scores.unsqueeze(0)
        # Remove Lower Triangle and Diagonal
        final_scale = iou * conf_matrix.triu(1)

        # batch*num_priors
        ae_loss = self.autoencoder(original.unsqueeze(0), loc[keep].unsqueeze(0))
        # ae_loss but in scores
        ae_grid = torch.zeros_like(final_scale)
        ae_grid[keep] = ae_loss
        ae_grid = ae_grid * final_scale
        return torch.sum(ae_grid)


class MyException(Exception):
    pass


class VarianceLoss(nn.Module):
    def __init__(self):
        super(VarianceLoss, self).__init__()
        pass

    def forward(self, original, predictions):
        # original is [batch_size, 3, img_h, img_w]

        # This is correct, because of the tranpose above
        # conf shape: torch.size(batch_size,num_priors,num_classes)
        # predictions is array of Dicts from detect
        # boxes=loc Shape: [num_priors, 5]

        resizeShape = list(original.shape)[-2:]

        # Assuming it has no batch
        priors_shape = list([a["conf"].size(0) for a in predictions])

        # These are concatenated, not Pad_Sequenced on purpose.
        loc = torch.cat([a["loc"] for a in predictions], dim=0)

        # batch, num_priors, i,j, with Padded sequence
        unnormalGaussian = pad_sequence(
            # Split results by shapes of the priors
            torch.split(
                # Unsqueeze to add fake batch [0] to remove fake batch
                gaussian.unnormalGaussian(maskShape=resizeShape, loc=loc.unsqueeze(0))[
                    0
                ],
                priors_shape,
            ),
            batch_first=True,
        )

        # masks: [batch, num_priors, mask_dim] with Padded Sequence
        masks = pad_sequence([a["masks"] for a in predictions], batch_first=True)

        # proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)
        proto = torch.stack([a["proto"] for a in predictions])

        # Dim: Batch, Anchors, i, j
        assembledMask = gaussian.lincomb(proto=proto, masks=masks)

        # Dim: Batch, Anchors, i, j
        attention = assembledMask * unnormalGaussian

        # conf shape: torch.size(batch_size,num_priors) #no num_classes
        # conf = torch.cat([a["conf"] for a in predictions], dim=0)
        # conf = torch.cat([a["conf"] for a in predictions], dim=0)
        conf = pad_sequence([a["conf"] for a in predictions], batch_first=True)
        # num_priors, 1
        # conf = conf[:, 1]  # Confidence in foreground. Dim Batch, Priors

        # if conf.shape[-1] != 2:
        # raise MyException("Wrong number of classes")

        # No need for masking, since pad_sequence already inputs zeros
        # Batch, Anchors, i,j
        maskConfidence = torch.einsum("abcd,ab->abcd", attention, conf)
        # logConf = torch.log(maskConfidence)

        # unsup: REMOVE CHANNEL DIMENSION HERE, since Pad_sequence is summed, it does nothing
        # Confidence in background
        # Dim: batch, h, w
        finalConf = 1 - torch.sum(
            maskConfidence ** 2
            / (torch.sum(maskConfidence, dim=1, keepdim=True) + 1e-5),
            dim=1,
        )
        # finalConf = 1 - torch.sum(F.softmax(logConf, dim=1) * maskConfidence, dim=1)

        # Resize to Original Image Size, add fake depth
        # unsup: Interpolation may be incorrect
        # Dim batch, img_h, img_w
        resizedConf = F.interpolate(
            finalConf.unsqueeze(1), resizeShape, mode="bilinear"
        )[:, 0]

        # unsup: AGGREGATING RESULTS BETWEEN BATCHES HERE
        # Dim, h, w
        totalConf = torch.sum(resizedConf, dim=0)

        # Dim 3, img_h, img_w
        weightedMean = torch.einsum("abcd,acd->bcd", original, resizedConf)

        # Dim: batch, 3, img_h, img_w
        squaredDiff = (original - weightedMean) ** 2

        # Batch,3,img_h, image w
        weightedDiff = torch.einsum("abcd,acd->abcd", squaredDiff, resizedConf)
        weightedVariance = weightedDiff / totalConf

        # Normalize by number of elements in original image
        return torch.sum(weightedVariance)

    # was normalized by /original.numel()


# class ScaledAutoencoderLoss(nn.Module):
# def __init__(self, img_h, img_w):
# super(ScaledAutoencoderLoss, self).__init__()
# pass

