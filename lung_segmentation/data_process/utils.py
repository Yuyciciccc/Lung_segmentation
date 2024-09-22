from .convert import *
from .get_image_dimension import *
from data_process.save_to_csv import *
from data_process.dataset import *
from data_process.metrics import *
import torch
import matplotlib.pyplot as plt


def custom_collate(batch):
    images = [item['image'] for item in batch]
    masks = [item['label'] for item in batch]
    # file_names = [item['file_name'] for item in batch]  # 获取文件名列表
    
    return torch.stack(images), torch.stack(masks)#, file_names

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def write_metrics_to_csv(csv_path, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    """
    将训练和验证的metrics写入CSV文件

    Parameters:
    - csv_path (str): CSV 文件路径
    - epoch (int): 当前回合数
    - train_loss (float): 训练损失
    - train_accuracy (float): 训练准确率
    - val_loss (float): 验证损失
    - val_accuracy (float): 验证准确率
    """
    fieldnames = ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']

    # 如果文件不存在，则创建表头
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # 打开文件以追加模式写入数据
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入当前回合的metrics
        writer.writerow({
            'Epoch': epoch,
            'Train Loss': train_loss,
            'Train Accuracy': train_accuracy,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_accuracy
        })

def read_csv_to_lists(csv_path):
    """
    读取 CSV 文件为四个数据列表

    Parameters:
    - csv_path (str): CSV 文件路径

    Returns:
    - tuple of lists: 包含 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy' 的四个列表
    """
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # 从每一行的字典中提取对应的值，并将其加入到相应的列表中
            train_losses.append(float(row['Train Loss']))
            train_accuracies.append(float(row['Train Accuracy']))
            val_losses.append(float(row['Validation Loss']))
            val_accuracies.append(float(row['Validation Accuracy']))

    return train_losses, train_accuracies, val_losses, val_accuracies




