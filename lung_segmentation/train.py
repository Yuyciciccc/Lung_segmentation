import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.ModelUnet import *
from data_process.utils import MyDataset, custom_collate
from torchvision.utils import save_image
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
from data_process.metrics import calculate_accuracy
from data_process.utils import adjust_learning_rate
# Use CUDA
device = torch.device('cuda')
weight_path = 'log/params/Unet.pth'
data_path = 'data/train.csv'
save_path = r'log/train'
transform = transforms.Compose([transforms.ToTensor()])


if __name__ == '__main__':
    dataset = MyDataset(data_path)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate)

    net = UNet().to(device)
    
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("Successfully loaded weights!")
    else:
        print("Weights not found. Training from scratch!")

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    num_epoch = 100
    all_train_losses = []  
    all_val_losses = []  
    all_train_accuracies = []  
    all_val_accuracies = []  

    initial_lr = 0.001
    lr = initial_lr

    for epoch in range(1, num_epoch + 1):   
        train_losses = []  
        train_accuracies = []  

        for i, (image, mask) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            image, mask = image.to(device), mask.to(device)
            
            out_img = net(image)
            train_loss = loss_fun(out_img, mask)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            _image = image[0]
            _mask = mask[0]
            _out_image = out_img[0]

            img = torch.stack([_image, _mask, _out_image], dim=0)
            save_image(img, f'{save_path}/epoch_{epoch}_train_{i}.png')
            train_losses.append(train_loss.item())

            # 计算准确率
            accuracy = calculate_accuracy(out_img, mask)
            train_accuracies.append(accuracy)
   
               
            

        # 计算并打印平均训练损失和准确率
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)

        tqdm.write(f'Epoch {epoch} - Average Training Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}')
        
        all_train_losses.append(avg_train_loss)  
        all_train_accuracies.append(avg_train_accuracy)  

        # 在验证集上评估模型
        net.eval()
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for i, (image, mask) in enumerate(tqdm(val_loader, desc='Validation')):
                image, mask = image.to(device), mask.to(device)
                out_img = net(image)
                val_loss = loss_fun(out_img, mask)
                val_losses.append(val_loss.item())

                _image = image[0]
                _mask = mask[0]
                _out_image = out_img[0]

                img = torch.stack([_image, _mask, _out_image], dim=0)
                save_image(img, f'{save_path}/epoch_{epoch}_vali_{i}.png')
                # 计算准确率
                accuracy = calculate_accuracy(out_img, mask)
                val_accuracies.append(accuracy)

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
        tqdm.write(f'Validation - Average Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}')
        all_val_losses.append(avg_val_loss)
        all_val_accuracies.append(avg_val_accuracy)

        # 在这里加入根据验证集性能调整超参数的逻辑
        # 这里使用简单的学习率调整作为示例
        if epoch > 1 and len(all_val_losses) >= 2 and all_val_losses[-1] > all_val_losses[-2]:
            # 如果验证集损失上升，减小学习率
            lr *= 0.9
            adjust_learning_rate(opt, lr)
            tqdm.write(f'Reducing learning rate to: {lr}')
        
        torch.save(net.state_dict(), f'log/params/Unet.pth')
        

    # 绘制训练和验证损失以及准确率曲线
    plt.figure(figsize=(12, 9))
    plt.plot(range(1, num_epoch + 1), all_train_losses, label='Train Loss')
    plt.plot(range(1, num_epoch + 1), all_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(r'Loss.png')
    plt.show()

    plt.figure(figsize=(12, 9))
    plt.plot(range(1, num_epoch + 1), all_train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epoch + 1), all_val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.savefig(r'Accuracy.png')
    plt.show()
