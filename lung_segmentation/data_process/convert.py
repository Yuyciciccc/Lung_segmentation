import tifffile
import imageio
import os
import numpy as np
from PIL import Image
import pydicom
import scipy.misc

def tif_to_png(tif_path, png_path):
    # 检查TIF文件是否存在
    if not os.path.isfile(tif_path):
        print(f"错误:未找到TIFF文件 '{tif_path}'。")
        return
    # 读取TIFF图像
    tif_image = tifffile.imread(tif_path)
    #   图像归一化，未归一化在保存时会存在缩放问题导致图像不正常
    normalized_image = (tif_image - np.min(tif_image)) / (np.max(tif_image) - np.min(tif_image))
    # 保存为PNG文件
    imageio.imwrite(png_path, (normalized_image * 255).astype(np.uint8))

def tif_to_png_dir(tif_dir,png_dir):
    tif_files=[os.path.join(tif_dir, file) for file in os.listdir(tif_dir) if file.endswith('.tif')]
    for i in range(len(tif_files)):
        filename=os.path.splitext(os.path.basename(tif_files[i]))[0]+".png"
        png_path=png_dir+'/'+filename
        tif_to_png(tif_files[i],png_path)
    
def pil_to_png(pil_path,png_path):
    image = Image.open(pil_path)
    # 保存为PNG格式
    image.save(png_path, format='PNG')

def pil_to_png_dir(pil_dir,png_dir):
    pil_files=[os.path.join(pil_dir, file) for file in os.listdir(pil_dir) if file.endswith('.pil')]
    for i in range(len(pil_files)):
        filename=os.path.splitext(os.path.basename(pil_files[i]))[0]+".png"
        png_path=png_dir+'/'+filename
        tif_to_png(pil_files[i],png_path)
# !
# def dcm_to_png(dcm_path,png_path):
#     dcm=pydicom.read_file(dcm_path)
#     image=dcm.pixel_array
#     scipy.misc.imsave(png_path,image)

# def dcm_to_png_dir(dcm_dir,png_dir):
#     dcm_files=[os.path.join(dcm_dir, file) for file in os.listdir(dcm_dir) if file.endswith('.dcm')]
#     for i in range(len(dcm_files)):
#         filename=os.path.splitext(os.path.basename(dcm_files[i]))[0]+'.png'
#         png_path=png_dir+'/'+filename
#         dcm_to_png(dcm_files[i],png_path)

if __name__=='__main__':
    pil_dir=''