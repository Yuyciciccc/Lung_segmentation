import tifffile
import imageio
import os
import numpy as np

def tif_to_png(tif_path, png_path):
    # 检查TIFF文件是否存在
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
    

