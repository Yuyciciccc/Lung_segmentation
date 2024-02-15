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
#todo 添加filename
    def __getitem__(self, index):
        image_path = os.path.join(self.data_info.iloc[index, 0])
        mask_path = os.path.join(self.data_info.iloc[index, 1])

        image = Image.open(image_path).convert('RGB')   
        mask = Image.open(mask_path).convert('RGB')   
        # print(f"Image path: {image_path}, Mask path: {mask_path}")
       
        image = self.transform(image)
        mask = self.transform(mask)
        # print(f"Image type: {type(image)}, Mask type: {type(mask)}")
        return {'image': image, 'mask': mask}

    def getdata(self):
        data = pd.read_csv(self.csv_file)
        images = [self.transform(Image.open(os.path.join(img)).convert('RGB')) for img in data['Image']]
        masks = [self.transform(Image.open(os.path.join(mask)).convert('RGB')) for mask in data['Mask']]
        return images, masks

