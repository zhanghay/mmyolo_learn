"""
# mmpretrain.MobileNetV2
from mmpretrain.models import MobileNetV2
import torch

self = MobileNetV2(
    widen_factor=1.,
    out_indices=(5, 6, 7),
    frozen_stages=-1,
    conv_cfg=None,
    norm_cfg=dict(type='BN'),
    act_cfg=dict(type='ReLU6'),
    norm_eval=False,
    with_cp=False,
    init_cfg=dict(
        type='Pretrained', checkpoint='down_model/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
    ))
self.eval()
inputs = torch.rand(1, 3, 32, 32)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))  # [192, 432, 1008] [160,320,1280]
"""

'''
# mmdet.RegNet
from mmdet.models import RegNet
import torch

self = RegNet(
    arch=dict(
        w0=88,
        wa=26.31,
        wm=2.25,
        group_w=48,
        depth=25,
        bot_mul=1.0),
    out_indices=(1, 2, 3), )
self.eval()
inputs = torch.rand(1, 3, 32, 32)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))  # [192, 432, 1008]
'''
'''
from mmdet.models import ResNet
import torch

self = ResNet(depth=18, out_indices=(1, 2, 3))  # 这一行，尽量所有参数和配置文件一样
self.eval()
inputs = torch.rand(1, 3, 32, 32)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))
'''
'''
# mmpretrain.SwinTransformer
from mmpretrain.models import SwinTransformer
import torch

self = SwinTransformer(
    arch='tiny',
    img_size=224,
    patch_size=4,
    in_channels=3,
    window_size=7,
    drop_rate=0.,
    drop_path_rate=0.1,
    out_indices=(1,2,3),
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
        prefix='backbone.'
    ))
self.eval()
inputs = torch.rand(1, 3, 224, 224)
level_outputs = self.forward(inputs)
for level_out in level_outputs:
    print(tuple(level_out.shape))  # [192, 432, 1008] [160,320,1280]
'''
'''
import torch
from mmyolo.models import PPYOLOECSPResNet
from mmyolo.utils import register_all_modules

# 注册所有模块
register_all_modules()

imgs = torch.randn(1, 3, 640, 640)
out_indices = (2, 3, 4)
model = PPYOLOECSPResNet(arch='P5', widen_factor=1.5, out_indices=out_indices)
out = model(imgs)
out_shapes = [out[i].shape for i in range(len(out_indices))]
print(out_shapes)
'''

from mmyolo.models import YOLOv5CSPDarknet
from mmyolo.models import YOLOv5PAFPN
import torch
model = YOLOv5CSPDarknet()
model.eval()
inputs = torch.rand(1, 3, 416, 416)
backbone_outputs = model(inputs)

model = YOLOv5PAFPN(in_channels=[256, 512, 1024])
model.eval()
neck_output = model(backbone_outputs)
for level_out in neck_output:
    print(tuple(neck_output.shape))
