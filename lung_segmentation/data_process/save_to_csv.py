import os
import random
import csv

def split_and_save_to_csv(image_dir, mask_dir, train_csv, test_csv,split_ratio=0.8):
    # 获取image文件夹中的所有文件路径
    image_files = file_name_path(image_dir,False,True)
    for i in range(len(image_files)):
        image_files[i]=os.path.join(image_dir,image_files[i])
    # 打乱顺序
    random.shuffle(image_files)
    # 计算分割点
    split_point = int(len(image_files) * split_ratio)
    # 分割为训练集和测试集
    train_images = image_files[:split_point]
    test_images = image_files[split_point:]

    # 构建对应的mask文件路径
    train_masks = [os.path.join(mask_dir, os.path.basename(image)) for image in train_images]
    test_masks = [os.path.join(mask_dir, os.path.basename(image)) for image in test_images]

    # 保存训练集到CSV文件
    save_to_csv(train_csv, train_images, train_masks)

    # 保存测试集到CSV文件
    save_to_csv(test_csv, test_images, test_masks)


def save_to_csv(csv_file, images, masks):

    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'Mask'])  # 写入表头

        # 逐行写入文件路径
        for image, mask in zip(images, masks):
            csv_writer.writerow([image, mask])


#   返回file_dir下所有的的sub_dir或者file
def file_name_path(file_dir,dir=True,file=False):

    for root,dirs,files in os.walk(file_dir):
        if len(dirs) and dir:
#            print("sub_dirs: ", dirs)
            return dirs
        if len(files) and file:
#            print("files: ", files)
            return files


#   输出目录下的子目录或文件
def print_file_name_path(file_dir,dir=True,file=False):

    result = file_name_path(file_dir, dir=dir, file=file)
    if(dir):
        print("sub_dirs: ")
    if(file):
        print("files: ")
    # 即每行只输出四个路径（起始，总数，步长）
    for i in range(0,len(result),4):
        #切片，i为起始索引，i+4为结束索引但不包括
        print(result[i:i+4])
    return result


#   全部导入
def save_file2_csv(file_dir, file_name):
    """
    将文件路径保存到CSV文件中,这是为了图像分割而设计的
    :param file_dir: 预处理数据的路径
    :param file_name: 输出CSV文件的名称

    #Note: 路径中/不用转义，\需要转义.windows系统中文件路径为\
    """
    # 打开CSV文件，准备写入数据
    out=open(file_name,'w')
    # 定义图像和掩膜的文件夹名称
    image="2d_images"
    mask="2d_masks"
    # 构建图像和掩膜的完整路径
    file_image_dir=(file_dir+'/'+image)
    file_mask_dir=(file_dir+'/'+mask)
    # 获取图像和掩膜文件的路径列表
    file_image_path=file_name_path(file_image_dir,False,True)
    file_mask_path=file_name_path(file_mask_dir,False,True)
    # 写入CSV文件的第一行，列名为"Image,Mask"
    out.write("Image,Mask"+'\n')
    # 遍历文件路径列表，写入每一行数据
    for i in range(len(file_image_path)):
        # 构建图像和掩膜文件的完整路径
        file_image=file_image_dir+'/'+file_image_path[i]
        file_mask=file_mask_dir+'/'+file_mask_path[i]
        # 将图像路径和掩膜路径写入CSV文件的一行
        out.writelines(file_image+','+file_mask+'\n')

def dataprocess(data_dir):    
    
    image_dir=data_dir+"/2d_images"
    mask_dir=data_dir+"/2d_masks"

    save_file2_csv(data_dir,r'data/data.csv')
    #   file_name_path只返回文件名
    image_files = file_name_path(image_dir,False,True)
    for i in range(len(image_files)):
        image_files[i]=os.path.join(image_dir,image_files[i])
    #   图像和标签文件名需一致
    mask_files=[os.path.join(mask_dir, os.path.basename(image)) for image in image_files]
    #   分割训练集和测试集
    train_csv=r'data/train.csv'
    test_csv=r'data/test.csv'
    split_and_save_to_csv(image_dir,mask_dir,train_csv,test_csv,split_ratio=0.8)










