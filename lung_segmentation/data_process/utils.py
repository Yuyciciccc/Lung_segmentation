from .convert import *
from .get_image_dimension import *
from data_process.save_to_csv import *
from data_process.dataset import *
from data_process.metrics import *
import torch

def custom_collate(batch):
    images = [item['image'] for item in batch]
    masks = [item['mask'] for item in batch]

    # 转换图像和掩码为张量
    images = torch.stack(images)
    masks = torch.stack(masks)

    return images, masks

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr