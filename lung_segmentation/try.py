# from data_process import *
# from data_process.save_to_csv import *

# img_dir=r'G:\06_CodeData\Database\Lung segmentation\2d_png_images'
# mask_dir=r'G:\06_CodeData\Database\Lung segmentation\2d_png_masks'
# train_csv=r'data\train2.csv'
# test_csv=r'data\test2.csv'
# split_and_save_to_csv(img_dir,mask_dir,train_csv,test_csv)

# import numpy as np
# from PIL import Image

# # 第一张图片和其对应的掩膜文件路径
# image1_path = r'log\Unet\test_mask\ID_0005_Z_0066.png'
# image2_mask_path = r'G:\06_CodeData\Database\Lung_segmentation\2d_png_masks\ID_0005_Z_0066.png'

# # 加载第一张图片
# image1 = Image.open(image1_path)

# # 加载第二张图片的掩膜
# image2_mask = Image.open(image2_mask_path).convert("L")

# # 将掩膜转换为 NumPy 数组
# image2_mask_np = np.array(image2_mask)

# # 根据掩膜找到裁剪边界
# nonzero_pixels = np.nonzero(image2_mask_np)
# min_x = np.min(nonzero_pixels[1])
# max_x = np.max(nonzero_pixels[1])
# min_y = np.min(nonzero_pixels[0])
# max_y = np.max(nonzero_pixels[0])

# # 裁剪第一张图片
# cropped_image1 = image1.crop((min_x, min_y, max_x, max_y))

# # 显示裁剪后的图像
# cropped_image1.show()

from matplotlib import pyplot as plt  
from PIL import Image 
plt.figure(figsize=(8, 5))  
plt.subplot(1, 1, 1)  
  
# 读取图像数据  
img_path = r'G:\06_CodeData\Database\Lung_segmentation\2d_png_images\ID_0071_Z_0072.png'  
img_data = Image.open(img_path)  # 读取图像并转换为灰度  
  
# 读取mask数据  
mask_path = r'log\unet\test_mask\ID_0071_Z_0072.png'  
mask_data = Image.open(mask_path).convert('L')  # 读取mask图像  
  
# 绘制图像  
plt.imshow(img_data, cmap='gray',vmin=-100, vmax=800)  # 绘制图像  
  
# 绘制mask轮廓  
# 如果遮罩图像是二值化的（即只有0和255），可以直接使用plt.contour  
# 如果不是，你可能需要首先进行阈值处理，将遮罩转换为二值图像  
contours = plt.contour(mask_data, colors='red')  # 绘制遮罩轮廓，假设遮罩是二值化的  
  
plt.axis('off')  
plt.tight_layout()  
plt.show()
