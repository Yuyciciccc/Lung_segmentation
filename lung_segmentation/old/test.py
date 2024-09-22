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
save_test_individual_path=r'log/Unet/test_mask'
save_path=r'log/Unet/test'
weight_path=r'log/params/Unet.pth'
loss_fun = nn.BCELoss()

if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print("Successfully load model")

test_csv=r'data/test.csv'
data_loader = DataLoader(MyDataset(test_csv), batch_size=1, collate_fn=custom_collate)
# 设置模型为评估模式
net.eval()

# 创建一个用于存储指标的容器
all_f1, all_recall, all_precision, all_dice, all_iou = [], [], [], [], []
all_test_losses = []  

# 进行测试
with torch.no_grad():
    for i, (image, mask,file_names) in enumerate(data_loader):
        image, mask = image.cuda(), mask.cuda()
        # 进行预测
        output = net(image)
        test_loss = loss_fun(output, mask)
        all_test_losses.append(test_loss.item())
        f1, recall, precision,dice,iou = calculate_metrics(output[0, 0], mask[0, 0])
        all_f1.append(f1)
        all_recall.append(recall)
        all_precision.append(precision)
        all_dice.append(dice)
        all_iou.append(iou)

        _image=image[0]
        _mask=mask[0]
        _output=output[0]
        save_image(output,f'{save_test_individual_path}/{file_names[0]}.png')
        img=torch.stack([_image,_mask,_output],dim=0)
        save_image(img,f'{save_path}/{file_names[0]}.png')

average_f1=np.mean(all_f1)
average_recall = np.mean(all_recall)
average_precision = np.mean(all_precision)
average_dice = np.mean(all_dice)
average_iou = np.mean(all_iou)
average_test_loss = np.mean(all_test_losses)

print(f'Average F1: {average_f1:.8f}')
print(f'Average Recall: {average_recall:.8f}')
print(f'Average Precision: {average_precision:.8f}')
print(f'Average Dice: {average_dice:.8f}')
print(f'Average IoU: {average_iou:.8f}')
print(f'Average Test Loss: {average_test_loss:.8f}')

