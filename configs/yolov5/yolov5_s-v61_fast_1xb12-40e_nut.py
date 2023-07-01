_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = '/home/hangyuan/nx/code/VisDA/deep-transfer-learning/UDA/pytorch1.0/DSAN/dataset/nut/without_nut_uva/'
class_name = ('MissingNut', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

anchors = [
           [(9, 9), (14, 12), (16, 16)],
           [(20, 17), (24, 19), (21, 25)],
           [(29, 24), (31, 32), (44, 37)]
           ]

max_epochs = 10
train_batch_size_per_gpu = 4
train_num_workers = 1

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='osplit/train.json',
        data_prefix=dict(img='o135/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='osplit/test.json',
        data_prefix=dict(img='o135/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'osplit/test.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa