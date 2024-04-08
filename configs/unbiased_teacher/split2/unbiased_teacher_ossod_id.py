_base_ = './_unbiased_teacher_base.py'
dataset_type = 'OSSODCocoDataset'

data = dict(
    train=dict(
        type='SemiDataset',
        ann_file_u=f'./dataset/DIOR/annotations_json_split1/DIOR_ID_split1.json',
    )
)

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='SetEpochInfoHook')
]

# # -------------------------model------------------------------
model = dict(
    type='UnbiasedTeacher_OSSOD',
    cfg=dict(
        vis_dir = './_vis_split2_id_mix'
    ),
    ossod_cfg=dict(
        cfb_len = 100,
        k = 5,
        beta_init=1,
    ),
    roi_head=dict(
        type='StandardRoIHeadMB',
        bbox_head=dict(
            type='Shared2FCBBoxHeadBase')
    )
)
