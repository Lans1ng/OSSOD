# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
"""
Base-Detector for semi-supervised learning
"""
import cv2
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

import mmcv

from mmcv.runner import load_checkpoint
from mmdet.models.builder import build_detector
from mmdet.core import bbox2roi

from mmdet_extension.core.utils.classes import COCO_CLASSES


class SemiBaseDetector(nn.Module):
    CLASSES = COCO_CLASSES

    def __init__(self,
                 ema_config=None,
                 ema_ckpt=None,
                 classes=None
                 ):
        if ema_config is not None:
            if isinstance(ema_config, str):
                ema_config = mmcv.Config.fromfile(ema_config)
            self.ema_model = build_detector(ema_config['model'])
            if ema_ckpt is not None:
                load_checkpoint(self.ema_model, ema_ckpt, map_location='cpu')
            self.ema_model.eval()
        else:
            self.ema_model = None
        if classes is not None:
            self.CLASSES = classes

    def forward(self, img, img_metas, return_loss=True, gen_embeddings=False, **kwargs):
        if return_loss:
            if 'img_unlabeled' in kwargs or 'is_label_data' in kwargs:
                return self.forward_train_semi(img, img_metas, **kwargs)
            else:
                return self.forward_train(img, img_metas, **kwargs)
        else:
            if gen_embeddings:
                return self.model_init(img, img_metas, **kwargs)
            else:
                return self.forward_test(img, img_metas, **kwargs)

    def model_init(self, imgs, img_metas, **kwargs):

        if 'proposals' in kwargs:
            kwargs['proposals'] = kwargs['proposals'][0]
        return self.embeddings_test(imgs[0], img_metas[0], **kwargs)
    
    def embeddings_test(self,
                    img,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    gt_bboxes=None,
                    gt_labels=None):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if gt_bboxes and gt_labels:
            gt_labels = [gt_label.cpu().numpy().reshape(-1) for gt_label in gt_labels][0]
            gt_bboxes = [torch.squeeze(gt_bbox,dim=0) for gt_bbox in gt_bboxes]
    
            classes = self.roi_head.bbox_head.num_classes
            classes_dict = {}
            classes_list = [i for i in range(classes)]
            for c in classes_list:
                classes_dict[c] = []

            rois = bbox2roi(gt_bboxes)
            bbox_feats = self.roi_head.bbox_roi_extractor(
                x[:self.roi_head.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred, bbox_feats_ = self.roi_head.bbox_head(bbox_feats)

            bbox_feats_ = [bbox_feat.cpu().numpy() for bbox_feat in bbox_feats_]
            for i, gt_label in enumerate(gt_labels):
                classes_dict[gt_label].append(bbox_feats_[i])
                
        return classes_dict

    @staticmethod
    def parse_loss(losses):
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                losses[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
        return losses

    @staticmethod
    def split_pos_ig(gt_bboxes, gt_labels, with_ig_label=False):
        gt_bboxes_pos, gt_bboxes_ig = [], []
        gt_labels_pos, gt_labels_ig = [], []
        for i, (bboxes, labels) in enumerate(zip(gt_bboxes, gt_labels)):
            ig_idx = labels < 0
            gt_bboxes_ig.append(bboxes[ig_idx])
            gt_bboxes_pos.append(bboxes[~ig_idx])
            gt_labels_ig.append(-2 - labels[ig_idx])
            gt_labels_pos.append(labels[~ig_idx])
        if with_ig_label:
            return gt_bboxes_pos, gt_labels_pos, gt_bboxes_ig, gt_labels_ig
        else:
            return gt_bboxes_pos, gt_labels_pos, gt_bboxes_ig

    def update_ema_model(self, momentum=0.99):
        model_dict = self.state_dict()
        new_dict = OrderedDict()
        for key, value in self.ema_model.state_dict().items():
            if key[7:] in model_dict.keys():
                new_dict[key] = (
                        model_dict[key[7:]] * (1 - momentum) + value * momentum
                )
            else:
                raise Exception("{} is not found in student model".format(key))
        self.ema_model.load_state_dict(new_dict)

    def cuda(self, device=None):
        """Since ema_model is registered as a plain object, it is necessary
        to put the ema model to cuda when calling cuda function."""
        if self.ema_model:
            self.ema_model.cuda(device=device)
        return super().cuda(device=device)

    def __setattr__(self, name, value):
        # not update ema_model in optimizer
        if name == 'ema_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def rescale_bboxes(self, bboxes, meta, bbox_transform):
        device = bboxes.device
        scale_factor = meta['scale_factor']
        if bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)
        bboxes = bboxes.cpu().numpy()
        for bf in bbox_transform:
            bboxes, _ = bf(bboxes, None)
        bboxes = torch.from_numpy(bboxes).float().to(device)
        return bboxes
