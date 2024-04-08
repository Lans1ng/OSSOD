# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Re-implementation: Unbiased teacher for semi-supervised object detection

There are several differences with official implementation:
1. we only use the strong-augmentation version of labeled data rather than \
the strong-augmentation and weak-augmentation version of labeled data.
"""
import numpy as np
import torch
import os

import cv2
import mmcv
from mmcv.runner.dist_utils import get_dist_info

from mmdet.utils import get_root_logger
from mmdet.models.builder import DETECTORS
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from mmdet_extension.models.detectors import SemiTwoStageDetector


@DETECTORS.register_module()
class UnbiasedTeacher(SemiTwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 # ema model
                 ema_config=None,
                 ema_ckpt=None,
                 # ut config
                 cfg=dict(),
                 ):
        super().__init__(backbone=backbone, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, neck=neck, pretrained=pretrained,
                         ema_config=ema_config, ema_ckpt=ema_ckpt)
        self.debug = cfg.get('debug', False)
        self.vis_dir = cfg.get('vis_dir', None)
        self.num_classes = self.roi_head.bbox_head.num_classes
        self.cur_iter = 0

        # hyper-parameter
        self.score_thr = cfg.get('score_thr', 0.7)
        self.weight_u = cfg.get('weight_u', 3.0)
        self.use_bbox_reg = cfg.get('use_bbox_reg', False)
        self.momentum = cfg.get('momentum', 0.996)

        # analysis
        self.image_num = 0
        self.pseudo_num = np.zeros(self.num_classes)
        self.pseudo_num_tp = np.zeros(self.num_classes)
        self.pseudo_num_gt = np.zeros(self.num_classes)

    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1,
    ):
        device = img.device
        self.image_num += len(img_metas_unlabeled)
        self.update_ema_model(self.momentum)
        self.cur_iter += 1
        self.analysis()
        # # ---------------------label data---------------------
        #先利用labeled image去训练student
        losses = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
        losses = self.parse_loss(losses)
        # # -------------------unlabeled data-------------------
        bbox_transform = []
        for img_meta in img_metas_unlabeled_1:
            bbox_transform.append(img_meta.pop('bbox_transform'))
        #利用ema model(Teacher)去做unlabled data的inference
        bbox_results = self.inference_unlabeled(
            img_unlabeled, img_metas_unlabeled, rescale=True
        )
        #然后经过一系列变换生成伪标签。
        gt_bboxes_pred, gt_labels_pred = self.create_pseudo_results(
            img_unlabeled_1, bbox_results, bbox_transform, device,
            gt_bboxes_unlabeled, gt_labels_unlabeled, img_metas_unlabeled  # for analysis
        )

#         if self.debug:
#             self.visual_offline(img_unlabeled_1, gt_bboxes_pred, gt_labels_pred, img_metas_unlabeled_1)
        if self.debug:
            self.visual_offline(img_unlabeled_1, 
                                gt_bboxes_pred, 
                                gt_labels_pred, 
                                img_metas_unlabeled_1)  
        #用这些伪标签去监督student模型的学习
        losses_unlabeled = self.forward_train(img_unlabeled_1, img_metas_unlabeled_1,
                                              gt_bboxes_pred, gt_labels_pred)
        losses_unlabeled = self.parse_loss(losses_unlabeled)
        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            if key.find('bbox') != -1:
                losses_unlabeled[key] = self.weight_u * val if self.use_bbox_reg else 0 * val
            else:
                losses_unlabeled[key] = self.weight_u * val
        losses.update({f'{key}_unlabeled': val for key, val in losses_unlabeled.items()})
        #pseduo_num平均一张图中的pseudo label的数量
        #pseudo_num(acc)，TP占所有pseudo label的比例
        extra_info = {
            'pseudo_num': torch.Tensor([self.pseudo_num.sum() / self.image_num]).to(device),
            'pseudo_num(acc)': torch.Tensor([self.pseudo_num_tp.sum() / self.pseudo_num.sum()]).to(device)
        }
        losses.update(extra_info)
        return losses

    def create_pseudo_results(self, img, bbox_results, box_transform, device,
                              gt_bboxes=None, gt_labels=None, img_metas=None):
        """using dynamic score to create pseudo results"""
        gt_bboxes_pred, gt_labels_pred = [], []
        _, _, h, w = img.shape
        use_gt = gt_bboxes is not None
        for b, result in enumerate(bbox_results):
            bboxes, labels = [], []
            if use_gt:
                gt_bbox, gt_label = gt_bboxes[b].cpu().numpy(), gt_labels[b].cpu().numpy()
                scale_factor = img_metas[b]['scale_factor']
                gt_bbox_scale = gt_bbox / scale_factor
            for cls, r in enumerate(result):
                label = cls * np.ones_like(r[:, 0], dtype=np.uint8)
                flag = r[:, -1] >= self.score_thr
                bboxes.append(r[flag][:, :4])
                labels.append(label[flag])
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes[-1]) > 0:
                    overlap = bbox_overlaps(bboxes[-1], gt_bbox_scale[gt_label == cls])
                    iou = overlap.max(-1)
                    self.pseudo_num_tp[cls] += (iou > 0.5).sum()
                self.pseudo_num_gt[cls] += (gt_label == cls).sum()
                self.pseudo_num[cls] += len(bboxes[-1])
            bboxes = np.concatenate(bboxes)
            labels = np.concatenate(labels)
            for bf in box_transform[b]:
                bboxes, labels = bf(bboxes, labels)
            gt_bboxes_pred.append(torch.from_numpy(bboxes).float().to(device))
            gt_labels_pred.append(torch.from_numpy(labels).long().to(device))
        return gt_bboxes_pred, gt_labels_pred

    def analysis(self):
        if self.cur_iter % 500 == 0 and get_dist_info()[0] == 0:
            logger = get_root_logger()
            info = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                             in zip(self.CLASSES, self.pseudo_num, self.pseudo_num_tp)])
            info_gt = ' '.join([f'{a}' for a in self.pseudo_num_gt])
            logger.info(f'pseudo pos: {info}')
            logger.info(f'pseudo gt: {info_gt}')
            
    def visual_offline(self, img, boxes_list, labels_list, img_metas, img_id=[],
                       boxes_ignore_list=None, gt_feats_score=None, gt_confidence_score=None):
        img_norm_cfg = dict(
            mean=np.array([123.675, 116.28, 103.53]), std=np.array([58.395, 57.12, 57.375])
        )
        out_root = self.vis_dir
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        if len(img_id) == 0:
            img_id = list(range(len(img)))
        for id in img_id:

            c,h,w = img[id].shape
#             print(h,w)
            vis_ori_image = True
            if vis_ori_image:
                boxes, labels = boxes_list[id], labels_list[id]
                device = boxes.device
                scale_factor = torch.from_numpy(img_metas[id]['scale_factor']).to(device)#tensor([1.1975, 1.1975, 1.1975, 1.1975], device='cuda:0')
                filename = img_metas[id]['filename']
                img_np = mmcv.imread(filename)
#                 print('ori_image_shape',img_np.shape)#(800, 800, 3)
#                 img_np = mmcv.imresize(img_np, (h,w))  # Reshape to img_np size
                if img_metas[id]['flip']:
                    img_np = mmcv.imflip(img_np, direction=img_metas[id]['flip_direction'])
                img_np = np.ascontiguousarray(img_np)
#                 print(type(boxes))
#                 print(type(scale_factor),scale_factor)
                boxes = boxes / scale_factor 
                
            else:
                img_np = img[id].permute(1, 2, 0).cpu().numpy()
                img_np = mmcv.imdenormalize(img_np, **img_norm_cfg)
                boxes, labels = boxes_list[id], labels_list[id]
#             feats_score, confidence_score = gt_feats_score[id], gt_confidence_score[id]
            for i, (box, label) in enumerate(zip(boxes, labels)):
                x1, y1, x2, y2 = np.ascontiguousarray([int(a.cpu().item()) for a in box])
#                 print(x1, y1, x2, y2)
                img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (157, 80, 136), 2)
                label_text = self.CLASSES[label]
#                 print('self.CLASSES',self.CLASSES)
#                 print(label)
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (157, 80, 136), 2)
                cv2.putText(img_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
            if boxes_ignore_list:
                boxes_ignore = boxes_ignore_list[id]
                for box in boxes_ignore:
                    x1, y1, x2, y2 = [int(a.cpu().item()) for a in box]
                    img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (44, 160, 44), 2)
            img_name = img_metas[id]['filename'].split('/')[-1]
            mmcv.imwrite(img_np, os.path.join(out_root, img_name))
