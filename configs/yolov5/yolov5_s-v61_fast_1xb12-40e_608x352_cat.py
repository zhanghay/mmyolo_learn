# åºäºè¯¥éç½®è¿è¡ç»§æ¿å¹¶éåé¨åéç½®
_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = './data/cat/' # æ°æ®éæ ¹è·¯å¾
class_name = ('cat', ) # æ°æ®éç±»å«åç§°
num_classes = len(class_name) # æ°æ®éç±»å«æ°
# metainfo å¿é¡»è¦ä¼ ç»åé¢ç dataloader éç½®ï¼å¦åæ æ
# palette æ¯å¯è§åæ¶åå¯¹åºç±»å«çæ¾ç¤ºé¢è²
# palette é¿åº¦å¿é¡»å¤§äºæç­äº classes é¿åº¦
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

# åºäº tools/analysis_tools/optimize_anchors.py èªéåºè®¡ç®ç anchor
anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]
# æå¤§è®­ç» 40 epoch
max_epochs = 10
# bs ä¸º 12
train_batch_size_per_gpu = 12
# dataloader å è½½è¿ç¨æ°
train_num_workers = 4

# å è½½ COCO é¢è®­ç»æé
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

model = dict(
    # åºå®æ´ä¸ª backbone æéï¼ä¸è¿è¡è®­ç»
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)
    ))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # æ°æ®éæ æ³¨æä»¶ json è·¯å¾
        ann_file='annotations/trainval.json',
        # æ°æ®éåç¼
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
    # æ¯é 10 ä¸ª epoch ä¿å­ä¸æ¬¡æéï¼å¹¶ä¸æå¤ä¿å­ 2 ä¸ªæé
    # æ¨¡åè¯ä¼°æ¶åèªå¨ä¿å­æä½³æ¨¡å
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # warmup_mim_iter åæ°éå¸¸å³é®ï¼å ä¸º cat æ°æ®ééå¸¸å°ï¼é»è®¤çæå° warmup_mim_iter æ¯ 1000ï¼å¯¼è´è®­ç»è¿ç¨å­¦ä¹ çåå°
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    # æ¥å¿æå°é´éä¸º 5
    logger=dict(type='LoggerHook', interval=5))
# è¯ä¼°é´éä¸º 10
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
