import os
import pickle
import matplotlib.pyplot as plt
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import mmcv
import cv2
from mmcv.runner.dist_utils import get_dist_info

from mmdet.utils import get_root_logger
from mmdet.models.builder import DETECTORS
from mmdet.core import bbox2roi
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from mmdet_extension.models.detectors import SemiTwoStageDetector
from mmdet_extension.core.visualization import imshow_det_bboxes

debug = True

def bbox2result(bboxes, labels, num_classes):

    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

    
def feat2result(feats, labels, num_classes):
    n, c = feats.shape
    if feats.shape[0] == 0:
        return [np.zeros((0, c), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [feats[labels == i, :] for i in range(num_classes)]
    
class CFB:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])
    
    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.append(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items
    
    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])
    
@DETECTORS.register_module()
class UnbiasedTeacher_OSSOD(SemiTwoStageDetector):
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
                 # ossod_config
                 ossod_cfg = dict(),
                 ):
        super().__init__(backbone=backbone, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, neck=neck, pretrained=pretrained,
                         ema_config=ema_config, ema_ckpt=ema_ckpt)
        self.debug = cfg.get('debug', False)
        self.vis_dir = cfg.get('vis_dir','_vis_test')
        self.total_epoch = cfg.get('total_epoch', 12)
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

        if self.training:
            self.unlabeled_list = [0 for i in range(self.num_classes)]
            self.cfb_len = ossod_cfg.get('cfb_len', 100)
            self.dist_type = ossod_cfg.get('dist_type', 'cos')
            self.CFB = CFB(self.num_classes, self.cfb_len) 

            self.beta_init = ossod_cfg.get('beta_init', 1) 
            self.beta_end = 2

            self.nbrs_list = [None for i in range(self.num_classes)]

            self.dist_means = [0 for i in range(self.num_classes)]
            self.dist_stds = [0 for i in range(self.num_classes)]
            self.dists = [0 for i in range(self.num_classes)]
            
    def set_epoch(self, epoch): 
        self.roi_head.cur_epoch = epoch 
        self.roi_head.bbox_head.cur_epoch = epoch
        self.cur_epoch = epoch
        
    def update(self):
        if self.cur_iter%1 == 0:
            for cls in range(self.num_classes):

                nbrs = NearestNeighbors(n_neighbors=int(self.cfb_len/20 + 1), algorithm='ball_tree', metric='minkowski')
                feats = normalize(self.CFB.retrieve(cls), norm='l2')
                self.nbrs_list[cls] = nbrs.fit(feats)
                scores, _ = nbrs.kneighbors(feats)   
                distances = np.mean(scores[:, 1:], axis=1)
                    
                self.dist_means[cls] = np.mean(distances)                
                self.dist_stds[cls] = np.std(distances)
                
                self.beta = self.beta_init + self.beta_end*self.cur_epoch/self.total_epoch 

                self.dists[cls] = self.dist_means[cls] + self.beta*self.dist_stds[cls]
        
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses, x
    
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

        losses, x = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
        losses = self.parse_loss(losses)

        with torch.no_grad():
            ema_model = self.ema_model.module
            
            gt_labels_ = []
            for gt_label in gt_labels:
                gt_labels_.extend(gt_label.cpu().tolist())
           
            rois = bbox2roi(gt_bboxes)
            bbox_roi_extractor = ema_model.roi_head.bbox_roi_extractor          
            _, feats_labeled = ema_model.roi_head._bbox_forward(x, rois)
            feats_labeled = [feat.cpu().tolist() for feat in feats_labeled]
            self.CFB.add(feats_labeled, gt_labels_)
            

        if all(len(self.CFB.store[class_id]) == self.cfb_len for class_id in range(self.CFB.total_num_classes)):
            pass
        else:
            print('CFB initialization: ',[len(self.CFB.store[i]) for i in range(self.CFB.total_num_classes)])
            return losses
        
        self.update()

        # # -------------------unlabeled data-------------------
        bbox_transform = []
        for img_meta in img_metas_unlabeled_1:
            bbox_transform.append(img_meta.pop('bbox_transform'))

        feats_unlabeled, bbox_results = self.inference_unlabeled(
            img_unlabeled, img_metas_unlabeled, rescale=True, return_feats=True
        )       
        gt_bboxes_pred, gt_labels_pred, feats_unlabeled_ = self.create_pseudo_results(
            img_unlabeled_1, feats_unlabeled, bbox_results, bbox_transform, device,
            gt_bboxes_unlabeled, gt_labels_unlabeled, img_metas_unlabeled  # for analysis
        )

        if self.debug:
            self.visual(img_unlabeled_1, 
                                gt_bboxes_pred, 
                                gt_labels_pred, 
                                img_metas_unlabeled_1)   
        
        losses_unlabeled, feats_unlabeled = self.forward_train(img_unlabeled_1, img_metas_unlabeled_1,
                                              gt_bboxes_pred, gt_labels_pred)
        _, feats_unlabeled_stu = self.roi_head._bbox_forward(x, rois)
        
        losses_unlabeled = self.parse_loss(losses_unlabeled)
        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            if key.find('bbox') != -1:
                losses_unlabeled[key] = self.weight_u * val if self.use_bbox_reg else 0 * val
            else:
                losses_unlabeled[key] = self.weight_u * val
        losses.update({f'{key}_unlabeled': val for key, val in losses_unlabeled.items()})
        extra_info = {
            'pseudo_num': torch.Tensor([self.pseudo_num.sum() / (self.image_num+1e-7)]).to(device),
            'pseudo_num(acc)': torch.Tensor([self.pseudo_num_tp.sum() / (self.pseudo_num.sum()+1e-7)]).to(device)
        }
        losses.update(extra_info)
        return losses
    
    @torch.no_grad() 
    def inference_unlabeled(self, img, img_metas, rescale=True, return_feats=True):
        ema_model = self.ema_model.module
        x = ema_model.extract_feat(img)
        proposal_list = ema_model.rpn_head.simple_test_rpn(x, img_metas)

        det_bboxes, det_labels= ema_model.roi_head.simple_test_bboxes(
            x, img_metas, proposal_list, ema_model.roi_head.test_cfg, rescale=rescale)
        det_bboxes_ = bbox2roi(det_bboxes)

        _, bbox_feats = ema_model.roi_head._bbox_forward(x, det_bboxes_)

        split_list = [bbox.size(0) for bbox in det_bboxes]
        bbox_feats = torch.split(bbox_feats, split_list, dim=0)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.num_classes)
            for i in range(len(det_bboxes))]

        feats_results = [
            feat2result(bbox_feats[i], det_labels[i], self.num_classes)
            for i in range(len(bbox_feats))]

        if return_feats:
            return feats_results, bbox_results
        else:
            return bbox_results


        
    def create_pseudo_results(self, img, feat_results, bbox_results, box_transform, device,
                              gt_bboxes=None, gt_labels=None, img_metas=None):
        """using dynamic score to create pseudo results"""
        gt_bboxes_pred, gt_labels_pred = [], []  
        gt_feats_score, gt_confidence_score = [], []
        _, _, h, w = img.shape
        use_gt = gt_bboxes is not None
        
        classes_list = [i for i in range(self.num_classes)]
        feats_dict = {c: [] for c in classes_list}

        for b, result in enumerate(bbox_results):
            feat_result = feat_results[b]
            bboxes, labels = [], []
            confidence_score_list, feats_score_list = [], []
            if use_gt:
                gt_bbox, gt_label = gt_bboxes[b].cpu().numpy(), gt_labels[b].cpu().numpy()
                scale_factor = img_metas[b]['scale_factor']
                gt_bbox_scale = gt_bbox / scale_factor   
            for cls, r in enumerate(result):
                nbrs_ = self.nbrs_list[cls]
                label = cls * np.ones_like(r[:, 0], dtype=np.uint8)
                flag = r[:, -1] >= self.score_thr
                
                feats = feat_result[cls][flag]
                r_ = r[flag][:, :4]
                label_ = label[flag]

                if len(feats) > 0:
                    if self.dist_type == 'cos':
                        feats = normalize(feats, norm='l2')
                    else:
                        pass
                        
                    scores, _ = nbrs_.kneighbors(feats)

                    score = np.mean(scores[:, 1:], axis=1)

                    flag_ = score <= self.dists[cls]
                    
                    bboxes.append(r_[flag_])
                    labels.append(label_[flag_])

                else:
                    bboxes.append(r_)
                    labels.append(label_)
        
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
        return gt_bboxes_pred, gt_labels_pred, feats_dict


    def analysis(self):
        if self.cur_iter % 400== 0 and get_dist_info()[0] == 0:
            logger = get_root_logger()
            info = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                             in zip(self.CLASSES, self.pseudo_num, self.pseudo_num_tp)])
            info_gt = ' '.join([f'{a}' for a in self.pseudo_num_gt])
            logger.info(f'pseudo pos: {info}')
            logger.info(f'pseudo gt: {info_gt}')  
          
    def visual(self, img, boxes_list, labels_list, img_metas, img_id=[],
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

            vis_ori_image = True
            if vis_ori_image:
                boxes, labels = boxes_list[id], labels_list[id]
                device = boxes.device
                scale_factor = torch.from_numpy(img_metas[id]['scale_factor']).to(device)
                filename = img_metas[id]['filename']
                img_np = mmcv.imread(filename)

                if img_metas[id]['flip']:
                    img_np = mmcv.imflip(img_np, direction=img_metas[id]['flip_direction'])
                img_np = np.ascontiguousarray(img_np)

                boxes = boxes / scale_factor 
                
            else:
                img_np = img[id].permute(1, 2, 0).cpu().numpy()
                img_np = mmcv.imdenormalize(img_np, **img_norm_cfg)
                boxes, labels = boxes_list[id], labels_list[id]

            for i, (box, label) in enumerate(zip(boxes, labels)):
                x1, y1, x2, y2 = np.ascontiguousarray([int(a.cpu().item()) for a in box])
                img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (157, 80, 136), 2)
                label_text = self.CLASSES[label]
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                vis_other= False

                img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (157, 80, 136), 2)
                cv2.putText(img_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if boxes_ignore_list:
                boxes_ignore = boxes_ignore_list[id]
                for box in boxes_ignore:
                    x1, y1, x2, y2 = [int(a.cpu().item()) for a in box]
                    img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (44, 160, 44), 2)
            img_name = img_metas[id]['filename'].split('/')[-1]
            mmcv.imwrite(img_np, os.path.join(out_root, img_name))
            
    def show_result(self,
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=4,
                font_size=20,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=(255, 255, 255),
            mask_color=mask_color,
            thickness=4,
            font_size=30,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

