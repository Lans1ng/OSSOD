_base_ = './_unbiased_teacher_base.py'

data = dict(
    train=dict(
        ann_file_u=f'./dataset/DIOR/annotations_json_split1/DIOR_ID_MIX_OOD_split1.json',
    ),
)

