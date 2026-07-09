import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import cv2
import torchio as tio
import torchvision.transforms as transforms
from PIL import Image


# 定义图像和掩码的预处理
PREPROCESSING_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量并缩放至[0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 对RGB图像进行标准化
])

PREPROCESSING_MASK_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量（缩放至[0, 1]）
])

# 训练数据增强
TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),  # 仅沿X轴进行随机翻转
])

class FUNDUSDataset(Dataset):
    def __init__(self, root_dir='', test_txt_dir='', augmentation=False):
        """
        初始化ISIC数据集
        :param root_dir: 图像文件夹路径
        :param test_txt_dir: 测试集文件的路径
        :param augmentation: 是否应用数据增强
        """
        self.root_dir = root_dir
        self.remove_test_path = test_txt_dir
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSFORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSFORMS

    def train_transform(self, image, label, p):
        """
        应用数据增强：翻转
        :param image: 输入图像
        :param label: 输入掩码
        :param p: 翻转的概率
        :return: 增强后的图像和掩码
        """
        # 添加一个深度维度，变成4D张量：形状为 (channels, depth,height, width)
        image = image.unsqueeze(1)  # 将 (3, 256, 256) -> (3, 1,256, 256)
        label = label.unsqueeze(1)  # 对掩码做同样的处理
        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(axes=(1), flip_probability=p),
        ])
        image = TRAIN_TRANSFORMS(image)
        label = TRAIN_TRANSFORMS(label)
        return image, label

    def get_file_names(self):
        """
        获取图像文件名列表
        :return: 文件名列表
        """
        all_file_names = glob.glob(os.path.join(self.root_dir, '*.png'))  

        # 如果有测试集的路径，读取测试集的文件
        '''
        with open(self.remove_test_path, 'r') as file:
            for line in file:
                test_file_name = line.strip()  
                test_file_names.add(test_file_name)
        '''
        # 根据测试集文件去除相应的图像文件
        filtered_file_names = [
            f for f in all_file_names
        ]
        return filtered_file_names

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def create_mask(shape):
        """
        创建一个全零的掩码
        :param shape: 掩码的形状
        :return: 生成的掩码
        """
        return torch.zeros(shape, dtype=torch.uint8)

    @staticmethod 
    def project_to_2d(mask):
        """
        将3D掩码投影到2D
        :param mask: 3D掩码
        :return: 2D投影
        """
        projection = torch.max(mask, dim=0)[0]
        return projection.numpy()

    @staticmethod
    def min_enclosing_circle(projection):
        """
        计算2D投影的最小包围圆
        :param projection: 2D掩码
        :return: 圆心坐标和半径
        """
        points = np.column_stack(np.where(projection > 0))
        points = points.astype(np.float32)
        print(points.shape)       
        (x, y), radius = cv2.minEnclosingCircle(points.astype(np.float32))
        center = (int(x), int(y))
        radius = int(radius)
        return center, radius

    @staticmethod
    def create_circle_mask_2d(shape, center, radius):
        """
        创建一个圆形掩码
        :param shape: 掩码形状
        :param center: 圆心
        :param radius: 半径
        :return: 圆形掩码
        """
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, thickness=-1)
        return mask

    @staticmethod
    def apply_circle_mask_to_3d(mask, circle_mask_2d):
        """
        将2D圆形掩码应用到3D掩码中
        :param mask: 3D掩码
        :param circle_mask_2d: 2D圆形掩码
        :return: 更新后的3D掩码
        """
        for i in range(mask.shape[0]):
            mask[i] = torch.from_numpy(circle_mask_2d)
        return mask

    def __getitem__(self, index):
        path = self.file_names[index]

        # 加载图像（RGB）
        img = Image.open(path).convert('RGB')  # 使用PIL加载图像，并转换为RGB模式
        
        # 获取对应的掩码文件路径
        mask_path = path.replace("imgs", "masks/all")  # 假设掩码在对应的 masks 文件夹中
        mask = Image.open(mask_path).convert('L')  # 加载为灰度图像

        # 应用预处理到图像和掩码
        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)
        img_tensor=img
        mask_tensor=mask
        p = np.random.choice([0, 1])

        img, mask = self.train_transform(img, mask, p)

        mask = mask.data
        img = img.data
        '''
        hists = []

        for channel in range(3):  # RGB三个通道
            channel_img = img_tensor[channel]  # 单通道图像 [H, W]
            # 只计算病变区域的像素
            channel_pixels = channel_img[mask_tensor[0] > 0.7]  # mask_tensor形状为[1, H, W]
            if len(channel_pixels) > 0:
                # 由于图像已经归一化到[-1, 1]，直接计算直方图
                hist = torch.histc(channel_pixels, bins=16, min=-1, max=1)
                hist = hist / hist.sum() if hist.sum() > 0 else torch.zeros(16)
            else:
                hist = torch.zeros(16)
            hists.append(hist)
        '''
        hists_2=[]
        for channel in range(3):  # RGB三个通道
            channel_img = img_tensor[channel]  # 单通道图像 [H, W]
            # 只计算病变区域的像素
            channel_pixels = channel_img[(mask_tensor[0] < 0.7) & (mask_tensor[0] >0.1)]  # mask_tensor形状为[1, H, W]
            if len(channel_pixels) > 0:
                # 由于图像已经归一化到[-1, 1]，直接计算直方图
                hist = torch.histc(channel_pixels, bins=16, min=-1, max=1)
                hist = hist / hist.sum() if hist.sum() > 0 else torch.zeros(16)
            else:
                hist = torch.zeros(16)
            hists_2.append(hist)
        
        # 拼接三个通道的直方图
        #hist = torch.cat(hists, dim=0)  # [48]
        hist2=torch.cat(hists_2, dim=0)
        #hist_combined = torch.cat((hist, hist2), dim=0)
        hist_combined=hist2
        ##print(img.shape)
        #print(mask.shape)
        #temp=mask.repeat(2,1,1,1)
        #print(temp.shape)
        return {'data': img.repeat(2, 1, 1, 1), 'label': mask.repeat(3, 1, 1, 1), 'hist': hist_combined}
