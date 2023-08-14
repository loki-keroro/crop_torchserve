import os
import torch
from nets.unet import Unet as unet
from cfgs.parameter import *


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cuda:1')
    model = unet(num_classes=NUM_CLASSES, backbone=BACKBONE)
    model.load_state_dict(checkpoint)  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


if __name__ == "__main__":
    # 利用trace把模型转化为pt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trained_model = TRAINED_MODEL  # cfg.TRAINED_MODEL表示训练好的pth所在的位置
    model = load_checkpoint(os.path.join(current_dir,trained_model))
    example = torch.rand(DATA_SIZE)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(os.path.join(current_dir,PT_MODEL))
    output = traced_script_module(torch.ones(DATA_SIZE))
    print(output)
