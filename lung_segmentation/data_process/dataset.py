import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, csv_file=None):
        self.csv_file = csv_file
        self.data_info = pd.read_csv(csv_file)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data_info) 

    def __getitem__(self, index):
        image_path = os.path.join(self.data_info.iloc[index, 0])
        mask_path = os.path.join(self.data_info.iloc[index, 1])

        image = Image.open(image_path).convert('RGB')   
        mask = Image.open(mask_path).convert('RGB')   
        # print(f"Image path: {image_path}, Mask path: {mask_path}")
       
        image = self.transform(image)
        mask = self.transform(mask)
        # print(f"Image type: {type(image)}, Mask type: {type(mask)}")
        
        # 使用 os.path.splitext 分割文件名和扩展名
        file_name, _ = os.path.splitext(os.path.basename(image_path))
        # 再次使用 os.path.splitext 分割文件名和 ".gz" 扩展名
        file_name, _ = os.path.splitext(file_name)

        return {'image': image, 'mask': mask, 'file_name': file_name}  # 添加文件名到返回字典中
    
    def getdata(self):
        data = pd.read_csv(self.csv_file)
        images = [self.transform(Image.open(os.path.join(img)).convert('RGB')) for img in data['Image']]
        masks = [self.transform(Image.open(os.path.join(mask)).convert('RGB')) for mask in data['Mask']]
        return images, masks

