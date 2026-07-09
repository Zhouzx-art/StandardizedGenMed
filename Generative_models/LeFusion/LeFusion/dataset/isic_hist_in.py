from torch.utils.data.dataset import Dataset
import os
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 定义图像和掩码的预处理
PREPROCESSING_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量并缩放至[0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 对RGB图像进行标准化
])

PREPROCESSING_MASK_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量（缩放至[0, 1]）
])

class ISICInDataset(Dataset):
    def __init__(self, root_dir='', test_txt_dir='', augmentation=False):
        """
        参数：
            root_dir (string): 包含所有图像和掩码的目录路径。
            test_txt_dir (string): 包含待排除测试图像文件列表的文件路径。
            augmentation (bool): 是否应用数据增强（目前没有实现，但可以根据需要添加）。
        """
        self.root_dir = root_dir
        self.remove_test_path = test_txt_dir
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSFORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSFORMS

    def get_file_names(self):
        """
        获取图像文件的路径，排除在remove_test_path文件中列出的测试图像。
        """
        #all_file_names = glob.glob(os.path.join(self.root_dir, './**/*.png'), recursive=True)
        all_file_names = glob.glob(os.path.join(self.root_dir, '*.png'))
        
        filtered_file_names = [
            f for f in all_file_names 
        ]
        return sorted(filtered_file_names)

    def __len__(self):
        """
        返回数据集中的图像数量。
        """
        return len(self.file_names)

    def __getitem__(self, index):
        """
        获取指定索引处的图像和对应的掩码。
        """
        img_path = self.file_names[index] 

        # 加载图像（RGB）
        img = Image.open(img_path).convert('RGB')
        
        # 获取对应的掩码文件路径
        mask_path = img_path.replace("imgs", "masks/all")
        
        # 加载掩码（灰度图）
        mask = Image.open(mask_path).convert('L')

        # 应用预处理到图像和掩码
        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)
        img = img.unsqueeze(1)  # 将 (3, 256, 256) -> (3, 1,256, 256)
        mask = mask.unsqueeze(1)  # 对掩码做同样的处理
        return {
            'GT': img,
            'GT_name': os.path.basename(img_path),  # 获取图像文件名
            'gt_keep_mask': mask,
        }

