import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from .sr_util import brats_transform_augment  # Assuming this is useful for transformation

def get_paths_from_png(path_data, path_gt):
    """
    Get sorted paths for images and masks from given directories.
    """
    assert os.path.isdir(path_data), '{:s} is not a valid directory'.format(path_data)
    assert os.path.isdir(path_gt), '{:s} is not a valid directory'.format(path_gt)

    data_png = glob.glob(path_data + "*.png")
    gt_png = glob.glob(path_gt + "*.png")

    assert data_png, '{:s} has no valid data PNG files'.format(path_data)
    assert gt_png, '{:s} has no valid GT PNG files'.format(path_gt)
    assert len(data_png) == len(gt_png), 'Images and masks are not paired!'

    return sorted(data_png), sorted(gt_png)

class ISIC(Dataset):
    def __init__(self, dataroot, img_size, split='train', data_len=-1):
        """
        Args:
            dataroot (str): Root directory for the images and masks.
            img_size (int): Target image size for resizing.
            split (str): Dataset split (train, val, test).
            data_len (int): Length of the dataset.
        """
        self.img_size = img_size
        self.data_len = data_len
        self.split = split
        
       
        img_root=dataroot+'/imgs/'+split+'/'
        gt_root=dataroot+'/masks/all/'+split+'/'

        self.img_png_path, self.gt_png_path = get_paths_from_png(img_root, gt_root)
        self.data_len = len(self.img_png_path)
        '''
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        '''
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
            
            #这里可以考虑把0,1,2映射到0,127,255
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),  # 将灰度图转换为 [0,1] 的张量
        ])
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index for fetching the sample.

        Returns:
            dict: Dictionary containing 'image', 'mask', and 'case_name'.
        """
        base_name = os.path.basename(self.img_png_path[index])
        case_name = os.path.splitext(base_name)[0]
        
        
        # Load image (RGB)
        img = Image.open(self.img_png_path[index]).convert('RGB')
        img = self.image_transform(img)
        
        # Load mask (grayscale)
        mask = Image.open(self.gt_png_path[index]).convert('RGB')  # 'L' mode for grayscale
        #mask = self.mask_transform(mask)
        mask= self.image_transform(mask)
        # Perform data augmentations if needed (same function as in BRATS dataset)
        #img, gt = brats_transform_augment([img, gt], split=self.split)

        #return {'image': img, 'mask': gt, 'case_name': case_name}
        #FD是真实图像
        return {'FD': img, 'LD': mask, 'case_name': case_name}
