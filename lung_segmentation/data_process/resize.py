import torch
import torch.nn.functional as F
from PIL import Image
import os
from torchvision import transforms

def resize_image(input_image_path, output_image_path, target_size):
    # 读取图像
    image = Image.open(input_image_path)
    # 定义 ToTensor 转换
    to_tensor = transforms.ToTensor()
    # 转换图像为 PyTorch tensor，并增加一个维度
    image_tensor = to_tensor(image).unsqueeze(0) 
    # 使用 interpolate 函数进行缩小，mode='bilinear' 表示使用双线性插值
    resized_image_tensor = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)
    # 转换 tensor 为 PIL 图像
    to_pil = transforms.ToPILImage()
    resized_image = to_pil(resized_image_tensor.squeeze(0))
    # 保存缩小后的图像
    resized_image.save(output_image_path)

def resize_image_dir(input_dir,output_dir,target_size):
    
    os.makedirs(output_dir, exist_ok=True)
    # 获取输入目录中的所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for image_file in image_files:
        # 构建输入和输出路径
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        # 调整图像大小
        resize_image(input_path, output_path, target_size)

if __name__=='__main__':
    input_dir=r'G:\06_CodeData\Database\Lung segmentation\2d_png_masks'
    output_dir=r'G:\06_CodeData\Database\Lung segmentation\2d_png_resize_masks'
    resize_image_dir(input_dir,output_dir,(64,64))



