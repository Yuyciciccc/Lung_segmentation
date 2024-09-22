import os
import csv
from PIL import Image
import numpy as np

# 假设分割图像文件夹
segmented_folder = r'log\Unet\test_mask'
output_folder = r'log\Unet\superposition'

# 读取包含原始图像文件名的 CSV 文件
csv_file = r'data\Unet_test_png.csv'
original_png_foleder=r'G:\06_CodeData\Database\Lung_segmentation\2d_png_images'
# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开 CSV 文件并读取原始图像路径
with open(csv_file, "r") as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        original_image_name = row[0]  # 假设路径保存在第一列
        segmented_image_name = os.path.basename(original_image_name)

        # 加载分割图像（假设分割图像是二值掩膜）
        segmented_image_path = os.path.join(segmented_folder, segmented_image_name).replace('/',"\\")
        segmented_image = Image.open(segmented_image_path).convert("L")  # 转换为灰度图像

        # 加载原始图像
        original_image_path = os.path.join(original_png_foleder,segmented_image_name)
        original_image = Image.open(original_image_path)

        masked_image_np = np.array(segmented_image)
        masked_image_np=np.where(masked_image_np>100,1,0)

        original_image_np=np.array(original_image)

        cropped_image_np=Image.fromarray(np.uint8(original_image_np)*np.uint8(masked_image_np))
        cropped_image_np.save(f'{output_folder}/{segmented_image_name}')


