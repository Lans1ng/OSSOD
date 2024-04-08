_base_ = './_unbiased_teacher_base.py'

data = dict(
    train=dict(
        type='SemiDataset',
        ann_file_u=f'./dataset/DIOR/annotations_json_split2/DIOR_ID_MIX_split2.json',
    )
)
