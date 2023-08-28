_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = './data/cat/'
class_name = ('cat',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)
max_epochs = 50
train_batch_size_per_gpu = 4
train_num_workers = 4
widen_factor: float = 1.0
# load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

channels = [192, 384, 768]

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.SwinTransformer',
        arch='tiny',
        img_size=224,
        patch_size=4,
        in_channels=3,
        window_size=7,
        drop_rate=0.,
        drop_path_rate=0.1,
        out_indices=(1, 2, 3),
        out_after_downsample=False,
        use_abs_pos_embed=False,
        interpolate_mode='bicubic',
        with_cp=False,
        frozen_stages=-1,
        norm_eval=False,
        pad_small_map=False,
        norm_cfg=dict(type='LN'),
        stage_cfgs=dict(),
        patch_cfg=dict(),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='down_model/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth',
            prefix='backbone.'
        )
    ),
    neck=dict(
        widen_factor=widen_factor,
        in_channels=channels,
        out_channels=channels
    ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,
            in_channels=channels,
            widen_factor=widen_factor
        ),
        prior_generator=dict(base_sizes=anchors)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
