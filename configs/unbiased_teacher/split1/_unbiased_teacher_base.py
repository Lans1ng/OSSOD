gpu = 1
score = 0.7
samples_per_gpu = 8
total_epoch = 12
test_interval = 6
save_interval = 12 #
classes = ['chimney', 'dam', 'Expressway-Service-area','Expressway-toll-station', 'vehicle', 'groundtrackfield', 'overpass','stadium', 'tenniscourt', 'trainstation']
data_root = './dataset/DIOR/'

# # -------------------------dataset------------------------------
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

image_size = (800, 800)
pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=image_size, ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AugmentationUT', use_re=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

pipeline_u_share = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
]

pipeline_u = [
    dict(type='Resize', img_scale=[(1333, 800)], keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

pipeline_u_1 = [
    dict(type='AddBBoxTransform'),
    dict(type='ResizeBox', img_scale=image_size, ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='AugmentationUT', use_re=True, use_box=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'bbox_transform'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        type='SemiDataset',
        ann_file=f'./dataset/DIOR/annotations_json_split1/DIOR_ID_labeled_split1.json',
        ann_file_u=f'./dataset/DIOR/annotations_json_split1/DIOR_ID_unlabeled_split1.json',
        pipeline=pipeline, pipeline_u_share=pipeline_u_share,
        pipeline_u=pipeline_u, pipeline_u_1=pipeline_u_1,
        img_prefix=data_root, img_prefix_u=data_root,
        classes=classes
    ),
    val=dict(
        type='CocoDataset',
        ann_file=f'./dataset/DIOR/annotations_json_split1/DIOR_test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file=f'./dataset/DIOR/annotations_json_split1/DIOR_test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))

# evaluation = dict(interval=test_interval, metric='bbox', by_epoch=False, classwise=True, only_ema=True)
evaluation = dict(interval=test_interval, metric='bbox', classwise=True, only_ema=True)
# # -------------------------schedule------------------------------
learning_rate = 0.02 * samples_per_gpu * gpu / 32
optimizer = dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[total_epoch]
)
runner = dict(type='SemiEpochBasedRunner', max_epochs=total_epoch)

checkpoint_config = dict(interval=save_interval)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
custom_hooks = [
    dict(type='NumClassCheckHook'),
#     dict(type='SetEpochInfoHook')
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
load_from = f'./work_dirs/split1/baseline_ssod/latest.pth'
workflow = [('train', 1)]

# # -------------------------model------------------------------
model = dict(
    type='UnbiasedTeacher',
    ema_config='./configs/baseline/ema_config/ut_ema_split1.py',
    ema_ckpt=load_from,
    cfg=dict(
        debug=False,
        score_thr=score,
        use_bbox_reg=False
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=len(classes),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            # loss_cls=dict(
            #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_cls=dict(
                type='CEFocalLoss', use_sigmoid=False, gamma=1.5, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_wrt_candidates=False,
                ignore_iof_thr=0.5),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            ig_weight=0.0,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
        )
    ))