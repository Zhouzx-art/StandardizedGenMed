import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset

'''
def grayscale_label_to_color(label):
    # 确保输入是灰度图
    if len(label.shape) == 3:
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    
    # 初始化为黑色背景
    color_label = np.ones((*label.shape, 3), dtype=np.uint8) * 0  # 白色背景 [255, 255, 255]

    # 掩码定义
    outer_mask = label == 255                    # 白色外圈
    inner_mask = (label >= 100) & (label < 150)  # 灰色内圈（允许一定容差）

    # 设置颜色（BGR格式）
    color_label[outer_mask] = [127, 127, 127]   # 外圈
    color_label[inner_mask] = [255, 255, 255]   # 内圈

    return color_label

'''

class MedicalDataset(Dataset):
    def __init__(self, image_dir, pt):
        super().__init__()
        self.data = []
        self.label = []
        self.prompt = pt
        # 遍历目录中的所有图像文件
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.npy')):  # 根据你的数据格式调整
                image_path = os.path.join(image_dir, filename)
                if "A_DATASET" in image_dir:
                    label_path = image_path.replace('imgs', 'masks/all')
                else:
                    label_path = image_path.replace('image', 'label').replace('data', 'label')
                self.data.append(image_path)
                self.label.append(label_path)

        assert len(self.data) == len(self.label), "Data len is not equal as label len."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label_path = self.label[idx]

        # 根据是否为Fundus选择不同的source处理方式
        if "Fundus" in label_path:
            source = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            source = cv2.resize(source, (512,512), interpolation=cv2.INTER_NEAREST)
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        else:
            source = cv2.resize(cv2.imread(label_path), (512,512), interpolation=cv2.INTER_NEAREST)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        # 统一的target处理
        target = cv2.imread(image_path)
        target = cv2.resize(target, (512,512))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        prompt = self.prompt

        return dict(jpg=target, txt=prompt, hint=source)

class MedicalDatasetGenerate(Dataset):
    def __init__(self, dir, pt):
        super().__init__()
        self.label = []
        self.prompt = pt
        files = os.listdir(dir)
        for file in files:
            self.label.append(os.path.join(os.path.abspath(dir), file))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label_path = self.label[idx]
        ##！！！！这里的Resize可能有错误！！！
        #source = cv2.resize(cv2.imread(label_path), (512,512))
        '''
        source = cv2.resize(cv2.imread(label_path), (512,512), interpolation=cv2.INTER_NEAREST)
        if "Fundus" in label_path:
            source = grayscale_label_to_color(source)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        '''
        if "Fundus" in label_path:
            source = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            source = cv2.resize(source, (512,512), interpolation=cv2.INTER_NEAREST)
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        else:
            source = cv2.resize(cv2.imread(label_path), (512,512), interpolation=cv2.INTER_NEAREST)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        
        prompt = self.prompt

        return dict(txt=prompt, hint=source)
    
if __name__ == "__main__":

    files="/DATA_EDS2/liwy/datasets/MedicalImage/White.txt"
    dir = '/DATA_EDS2/liwy/datasets/MedicalImage/synthetic-label/race/Asian'

    dataset = MedicalDatasetGenerate(dir)
    print(len(dataset))

    item = dataset[1014]
    # jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    # print(jpg.shape)
    print(hint.shape)



