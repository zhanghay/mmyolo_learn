import torch
from mmdet.models import DyHead
from mmyolo.utils import register_all_modules
from mmyolo.models import YOLOv5CSPDarknet
from mmyolo.models import YOLOv5PAFPN
from mmdet.models import NASFPN
from mmdet.models import ChannelMapper

channels = [256, 512, 1024]
inputs = torch.rand(1, 3, 416, 416)
deepen_factor = 0.33
widen_factor = 0.5
model = YOLOv5CSPDarknet(
    deepen_factor=deepen_factor,
    widen_factor=widen_factor,
    act_cfg=dict(type='SiLU', inplace=True))
model.eval()

backbone_outputs = model.forward(inputs)

neck1 = YOLOv5PAFPN(
    deepen_factor=deepen_factor,
    widen_factor=widen_factor,
    in_channels=[256, 512, 1024],
    out_channels=[256, 512, 1024],
    num_csp_blocks=3,
    act_cfg=dict(type='SiLU', inplace=True)
)
'''
(1, 128, 52, 52)
(1, 256, 26, 26)
(1, 512, 13, 13)
'''
neck1.eval()
neck1_outputs = neck1.forward(backbone_outputs)

neck2 = ChannelMapper(
    in_channels=[128, 256, 512],
    out_channels=128,
)
'''
(1, 128, 52, 52)
(1, 128, 26, 26)
(1, 128, 13, 13)
'''
neck2.eval()
neck2_out = neck2.forward(neck1_outputs)

neck3 = NASFPN(
    in_channels=[128, 128, 128],
    out_channels=256,
    num_outs=5,
    stack_times=1
)
'''
(1, 256, 52, 52)
(1, 256, 26, 26)
(1, 256, 13, 13)
'''
neck3.eval()
neck3_out = neck3.forward(neck2_out)

for level_out in neck3_out:
    print(tuple(level_out.shape))
