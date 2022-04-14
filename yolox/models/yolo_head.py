#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

import math

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        nID=None,                   # TODO: reid
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): wheather apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.nID = nID                                              # TODO: ReID. total ids in dataset
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)      # TODO: ReID. embedding scale

        self.cls_convs = nn.ModuleList()        # cls conv layer
        self.reg_convs = nn.ModuleList()        # reg conv layer
        self.cls_preds = nn.ModuleList()        # cls pred layer
        self.reg_preds = nn.ModuleList()        # reg pred layer
        self.obj_preds = nn.ModuleList()        # obj pred layer
        self.stems = nn.ModuleList()            # stems

        # TODO: reid head
        self.reid_convs = nn.ModuleList()        # cls conv layer
        self.reid_preds = nn.ModuleList()        # cls pred layer
        self.emb_dim = 128                      # dimension of reid embedding
        self.reid_classifier = nn.Linear(self.emb_dim, self.nID)        # TODO: ReID classifier, only used when training
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))                # TODO: For Uncertainty loss
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))                 # TODO: For Uncertainty loss
        # self.s_det = nn.Parameter(3 * torch.ones(1))                # TODO: For Uncertainty loss
        # self.s_id = nn.Parameter(2 * torch.ones(1))                 # TODO: For Uncertainty loss
        self.settings = {}

        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):       # iteration over levels of output features
            self.stems.append(      # 1 BaseConv layer
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(      # 2 BaseConv layers
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(      # 2 BaseConv layers
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(      # 1 Conv2d layer, output channel is 'self.n_anchors * self.num_classes'
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(      # 1 Conv2d layer, output channel is '4'
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(      # 1 Conv2d layer, output channel is 'self.n_anchors * 1'
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            # TODO: reid head: reid_convs (2 * 3x3 Conv) + reid_preds (1 * 1x1 Conv)
            self.reid_convs.append(
                nn.Sequential(      # 2 BaseConv layers
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reid_preds.append(      # 1 Conv2d layer, output channel is 'self.emb_dim'
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.emb_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.expanded_strides = [None] * len(in_channels)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)      # TODO: ReID loss. CrossEntropyLoss, ignore -1 ids (e.g. no id information)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):     # xin: 256/8, 512/16, 1024/32
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        '''iteration over levels of output feature map: 256/8, 512/16, 1024/32'''
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            reid_x = x      # TODO: reid branch

            cls_feat = cls_conv(cls_x)                  # cls_conv: 256/8, 512/16, 1024/32
            cls_output = self.cls_preds[k](cls_feat)    # cls_preds: [batchsize, clsss_num, H/8, W/8]/8, /16, /32

            reg_feat = reg_conv(reg_x)                  # reg_conv: 256/8, 512/16, 1024/32
            reg_output = self.reg_preds[k](reg_feat)    # reg_preds: [batchsize, 4, W/8, H/8]/8, /16, /32
            obj_output = self.obj_preds[k](reg_feat)    # obj_preds: [batchsize, 1, W/8, H/8]/8, /16, /32

            # TODO: output of reid branch
            reid_feat = self.reid_convs[k](reid_x)                    # reid_conv: 256/8, 512/16, 1024/32
            reid_output = self.reid_preds[k](reid_feat)     # reid_preds: [batchsize, embedding_dim, H/8, W/8]/8, /16, /32

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output, reid_output], 1)     # cat order: reg(4), obj(1), cls(1), reid(128)
                output, grid = self.get_output_and_grid(                # [1, 6, H/s, W/s] ==> [1, 6, H/s, W/s]
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])      # list, H/s loops of range(W/s)
                y_shifts.append(grid[:, :, 1])      # list, W/s loops of range(H/s)
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                # TODO: reid output. obj_output, cls_output use sigmoid --> obj_output, cls_output use sigmoid, reid_output
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid(), reid_output], 1     # obj_output, cls_output use sigmoid
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            # TODO: no need to modify for reid
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes + self.emb_dim     # number of channel, reg(4) + obj(1) + cls
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)     # [1, 1, H/s, W/s, 2]
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)    # [batchsize, 5+n_cls+emb_dim, H/s, W/s] --> [batchsize, n_anchors, 5+n_cls+emb_dim, H/s, W/s]
        output = output.permute(0, 1, 3, 4, 2).reshape(         # [batchsize, n_anchors, 5+n_cls+emb_dim, H/s, W/s] --> [bs, n_anchors * H/s* W/s, 5+n_cls+emb_dim]
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)      # [1, 1, H/s, W/s, 2] --> [1, H/s * W/s, 2]
        output[..., :2] = (output[..., :2] + grid) * stride     # offset w.r.t top-left corner coordinate of grid cell
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride     # w, h
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        # get preds from outputs
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:6]  # [batch, n_anchors_all, n_cls]
        id_preds = outputs[:, :, 6:]  # [batch, n_anchors_all, emb_dim]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)   # [1, n_anchors_all]
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        id_targets = []           # TODO: reid_targets
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))       # [matched_anchor, class_number]
                reg_target = outputs.new_zeros((0, 4))                      # [matched_anchor, 4]
                l1_target = outputs.new_zeros((0, 4))                       # [n_anchors, 1]
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                id_target = outputs.new_zeros((0))                          # TODO: ReID. [matched_anchor]
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # get target for loss, 'per_image' is used for assignment
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]       # [matched_anchor, 4]
                gt_classes = labels[batch_idx, :num_gt, 0]                  # [matched_anchor]
                gt_ids = labels[batch_idx, :num_gt, 5]                      # TODO: ReID target, [matched_anchor]

                bboxes_preds_per_image = bbox_preds[batch_idx]              # [n_anchors_all, 4]

                # assignment between gts and anchors (e.g. positive anchors to optimize)
                try:
                    (
                        gt_matched_classes,                 # [matched_anchor], class of matched anchors
                        gt_matched_ids,                     # TODO: ReID. [matched_anchor], id of matched anchors
                        fg_mask,                            # [n_anchors], .sum()=matched_anchor, to mask out unmatched anchors
                        pred_ious_this_matching,            # [matched_anchor], IoU of matched anchors
                        matched_gt_inds,                    # [matched_anchor], index of gts for each matched anchor
                        num_fg_img,                         # [1], matched_anchor
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        gt_ids,
                    )
                except RuntimeError:
                    logger.info(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    print("OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size.")
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,     # [matched_anchor], class of matched anchors
                        gt_matched_ids,  # TODO: ReID. [matched_anchor], id of matched anchors
                        fg_mask,                # [n_anchors], .sum()=matched_anchor, to mask out unmatched anchors
                        pred_ious_this_matching,    # [matched_anchor], IoU of matched anchors
                        matched_gt_inds,        # [matched_anchor], index of gts for each matched anchor
                        num_fg_img,             # [1], matched_anchor
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        gt_ids,
                        "cpu",
                    )
                
                
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                # get target for optimization. Because of multiple optisive strategy, each gt has multiple anchors to optimize
                # so the number of targets is matched_anchor, other than number of gt
                cls_target = F.one_hot(     # https://github.com/Megvii-BaseDetection/YOLOX/issues/949
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)               # [matched_anchor, class_number]
                # We would like to encode the iou information into the target, to relieve the misalignment
                # between the classification and regression prediction.
                obj_target = fg_mask.unsqueeze(-1)                      # [n_anchors] --> [n_anchors, 1]
                reg_target = gt_bboxes_per_image[matched_gt_inds]       # [matched_anchor, 4]
                id_target = gt_matched_ids                              # [matched_anchor]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)              # cls target
            reg_targets.append(reg_target)              # reg target
            obj_targets.append(obj_target.to(dtype))    # obj target
            id_targets.append(id_target)                # TODO: ReID. id target
            fg_masks.append(fg_mask)                    # fg_mask
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)     # [matched_anchor, 1]
        reg_targets = torch.cat(reg_targets, 0)     # [matched_anchor, 4]
        obj_targets = torch.cat(obj_targets, 0)     # [all_anchor, 1]
        id_targets = torch.cat(id_targets, 0)       # TODO: ReID. [matched_anchor]
        id_targets = id_targets.to(torch.int64)     # TODO: ReID. [matched_anchor], float16 to int
        fg_masks = torch.cat(fg_masks, 0)           # [all_anchor]
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        # compute loss
        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)        # [matched_anchor, 4]
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)            # [all_anchor, 1]
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets     # [matched_anchor, 1]
            )
        ).sum() / num_fg
        # TODO: ReID. compute id loss
        id_preds = id_preds.view(-1, self.emb_dim)[fg_masks]                    # [matched_anchor, emb_dim]
        id_preds = self.emb_scale * F.normalize(id_preds)                       # [matched_anchor, emb_dim]
        id_preds = self.reid_classifier(id_preds).contiguous()                  # [matched_anchor, nID]
        loss_id = self.IDLoss(id_preds, id_targets)

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0

        # loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1      # TODO: original loss (only detection)
        # loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + 0.5 * loss_id        # TODO: ReID. set weight of loss_id to 0.5
        # loss = loss_id          # TODO: only train id head

        # TODO: ReID. Uncertainty Loss
        # print("self.s_det:", self.s_det, "self.s_id:", self.s_id)           # for debug (0114)
        det_loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        id_loss = loss_id
        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        self.settings.update({'s_det': self.s_det, 's_id': self.s_id})

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_id,                # TODO: ReID. return id loss
            loss_l1,
            num_fg / max(num_gts, 1),
            self.settings
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        gt_ids,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        img_size = imgs.shape[2:]
        # fg_mask: [all_anchors]
        # is_in_boxes_and_center: [gt_num, matched_anchors]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
            img_size
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]        # [matched_anchor, 4]
        cls_preds_ = cls_preds[batch_idx][fg_mask]                      # [matched_anchor, 1]
        obj_preds_ = obj_preds[batch_idx][fg_mask]                      # [matched_anchor, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]           # [1], matched_anchor

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)     # [gt_num, matched_anchor]

        gt_cls_per_image = (                # [gt_num, matched_anchor, class_num]
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)         # [gt_num, matched_anchor]

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (      # [gt_num, matched_anchor, 1]
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()         # [gt_num, matched_anchor, 1]
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()       # [gt_num, matched_anchor, 1]
            )
            pair_wise_cls_loss = F.binary_cross_entropy(        # [gt_num, matched_anchor]
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,     # [k_anchors]
            gt_matched_ids,         # [k_anchors]
            pred_ious_this_matching,    # [k_anchors]
            matched_gt_inds,        # [k_anchors]
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, gt_ids, num_gt, fg_mask)          # assignment strategy 3
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,             # [k_anchors]
            gt_matched_ids,                 # [k_anchors]
            fg_mask,                        # [all_anchors]
            pred_ious_this_matching,        # [k_anchors]
            matched_gt_inds,                # [k_anchors]
            num_fg,                         # k_anchors
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        img_size
    ):
        """assignment strategy 1: anchors whose center is inside corresponding gt_bbox"""
        expanded_strides_per_image = expanded_strides[0]        # [n_anchors_all]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image   # shift on image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image   # shift on image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )   # [n_anchor] -> [n_gt, n_anchor]
        #  gt_bboxes to tlbr
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )   # [n_gt, n_anchor]
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )   # [n_gt, n_anchor]
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )   # [n_gt, n_anchor]
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )   # [n_gt, n_anchor]

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center
        """assignment strategy 2: anchors whose center is inside the 5^2 area centered at gt_bbox center"""
        center_radius = 2.5
        # clip center inside image
        gt_bboxes_per_image_clip = gt_bboxes_per_image[:, 0:2].clone()
        gt_bboxes_per_image_clip[:, 0] = torch.clamp(gt_bboxes_per_image_clip[:, 0], min=0, max=img_size[1])
        gt_bboxes_per_image_clip[:, 1] = torch.clamp(gt_bboxes_per_image_clip[:, 1], min=0, max=img_size[0])
        # tlbr of gt_bboxes
        gt_bboxes_per_image_l = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers (combine 2 assignment strategy)
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all        # [n_anchor]

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]       # [n_labels, n_anchor]
        )
        del gt_bboxes_per_image_clip
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, gt_ids, num_gt, fg_mask):
        # strategy 3: Dynamic K, simplified SimOTA
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]        # gt classes for matched anchors
        gt_matched_ids = gt_ids[matched_gt_inds]                # TODO: ReID, gt ids for matched anchors
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, gt_matched_ids, pred_ious_this_matching, matched_gt_inds
