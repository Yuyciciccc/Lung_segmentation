from models.ModelUnet import *
import os
import torch
from torch.utils.data import DataLoader
from models.ModelUnet import *
from data_process.utils import MyDataset, custom_collate
from torchvision.utils import save_image
from data_process.utils import calculate_metrics
import numpy as np

net=UNet().cuda()
save_path=r'log/test'
weight_path=r'log/params/Unet.pth'

if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print("Successfully load model")

test_csv=r'data/test.csv'
data_loader = DataLoader(MyDataset(test_csv), batch_size=1, collate_fn=custom_collate)
# 设置模型为评估模式
net.eval()

# 创建一个用于存储指标的容器
all_f1, all_recall, all_precision, all_dice, all_iou = [], [], [], [], []
# 进行测试
with torch.no_grad():
    for i, (image, mask) in enumerate(data_loader):
        image, mask = image.cuda(), mask.cuda()
        # 进行预测
        output = net(image)

        f1, recall, precision,dice,iou = calculate_metrics(output[0, 0], mask[0, 0])
        all_f1.append(f1)
        all_recall.append(recall)
        all_precision.append(precision)
        all_dice.append(dice)
        all_iou.append(iou)

        _image=image[0]
        _mask=mask[0]
        _output=output[0]

        img=torch.stack([_image,_mask,_output],dim=0)
        save_image(img,f'{save_path}/{i}.png')

average_f1=np.mean(all_f1)
average_recall = np.mean(all_recall)
average_precision = np.mean(all_precision)
average_dice = np.mean(all_dice)
average_iou = np.mean(all_iou)

print(f'Average F1: {average_f1:.8f}')
print(f'Average Recall: {average_recall:.8f}')
print(f'Average Precision: {average_precision:.8f}')
print(f'Average Dice: {average_dice:.8f}')
print(f'Average IoU: {average_iou:.8f}')

