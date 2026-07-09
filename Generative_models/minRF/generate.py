import torch
import os
from PIL import Image
from tqdm import tqdm
from dit import DiT_Llama
from dataset import ORIGADataset  # 假设你已经有这个 dataset 类
from torchvision import transforms
from rf import RF
import numpy as np
import argparse
import time
import shutil
# 定义用于将图像范围从[-1,1]转换到[0,1]的函数
def normalize_img(t):
    return t * 0.5 + 0.5

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default="None",
                        help='Path to the pretrained model checkpoint (.pt file) to load model weights.')
    parser.add_argument('--output_dir', type=str, default="None",
                        help='Directory where generated fake images will be saved.')
    parser.add_argument('--mask_output_dir', type=str, default="None",
                        help='Directory where original masks will be copied to (with renamed filenames).')
    parser.add_argument('--dataset_name', type=str, default="None",
                        help='Name of the dataset, used for naming output image files.')
    parser.add_argument('--imgs_path', type=str, default="None",
                        help='Path to the directory containing input images.')
    parser.add_argument('--masks_path', type=str, default="None",
                        help='Path to the directory containing corresponding masks.')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='Batch size for generation during inference.')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch size for the DiT model.')
    parser.add_argument('--start_num', type=int, default=8,
                        help='Starting number for generating sequential image filenames.')
    return parser.parse_args()

args = get_args()
# 设置参数
input_size = 256
channels = 3
batch_size = 8
val_batch_size = args.val_batch_size
sample_steps = 50  # 生成图像的步数

# 1. 指定使用的 GPU（例如 GPU 7）
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
# 2. 初始化模型
model = DiT_Llama(
        in_channels=channels,
        input_size=input_size,
        patch_size=args.patch_size,
        dim=1024,                    #隐藏层之前为512，1024
        n_layers=10,
        n_heads=8,
    )  # 将模型移动到指定的 GPU

# 加载训练好的模型权重
#model.load_state_dict(torch.load(args.pretrain))  
checkpoint = torch.load(args.pretrain, map_location='cuda')  #也可以用 'cuda' 但默认稳妥点
model.load_state_dict(checkpoint['model'])  # 加载模型权重
model = torch.nn.DataParallel(model).to(device)

# 创建 RF 对象
rf = RF(model)


# 数据预处理
image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    
    #这里可以考虑把0,1,2映射到0,127,255
mask_transform_in = transforms.Compose([
    transforms.Resize((32, 32),Image.NEAREST), 
    transforms.ToTensor(),  # 将灰度图转换为 [0,1] 的张量
])

# 3. 加载验证集
val_dataset = ORIGADataset(
    images_dir=args.imgs_path,
    masks_dir=args.masks_path,
    transform=image_transform,
    mask_transform=mask_transform_in
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,drop_last=False)

# 4. 预测生成并保存图片
#imgs_RF_all_1，表示不丢弃的版本，1，2，3为patch=8
output_dir = args.output_dir  # 生成图像的存储目录
os.makedirs(output_dir, exist_ok=True)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

model.eval()

with torch.no_grad():
    for i, (x, mask) in tqdm(enumerate(val_loader), desc="Generating Images"):
        start_time=time.time()
        x, mask = x.to(device), mask.to(device)  # 将数据移到指定的 GPU
        current_batch_size = x.shape[0]
        # 初始化噪声，准备生成图像
        init_noise = torch.randn(current_batch_size, channels, input_size, input_size).to(device)  # 移动到指定的 GPU
        null_cond = torch.zeros_like(mask).to(device)   # 例如：与 val_mask 形状相同的零张量
        null_cond.fill_(0.1)
        # 生成假图像
       
        fake_images = rf.sample(init_noise, mask, null_cond=None,sample_steps=sample_steps)[-1]  # 取采样的最后一帧结果
        
        # 处理生成的假图像
        fake_images = normalize_img(fake_images).clamp(0, 1).cpu()  # 归一化并裁剪图像
       
        end_time=time.time()-start_time
        print("生产图片耗时：",end_time)
        # 保存假图像

        
        for idx in range(fake_images.size(0)):
            # 将 mask 的文件名和生成的假图像对应
            mask_filename = val_dataset.image_filenames[i *args.val_batch_size + idx]
            png_num=i *args.val_batch_size + idx
            new_name = f"{args.dataset_name}_{png_num+args.start_num:04d}_0000.png"
            mask_new_name=f"{args.dataset_name}_{png_num+args.start_num:04d}.png"
            fake_image = fake_images[idx].permute(1, 2, 0).numpy()  # 转换为 HWC 格式
            fake_image = (fake_image * 255).astype(np.uint8)
            #if fake_image.shape[-1] == 1:
            #    fake_image = fake_image.squeeze(-1)  # shape: [H, W]
            # 保存生成的假图像，文件名与 mask 文件名一致
            fake_image_pil = Image.fromarray(fake_image)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fake_image_pil.save(os.path.join(output_dir, os.path.basename(new_name)))
            if not os.path.exists(args.mask_output_dir):
                os.makedirs(args.mask_output_dir)
            original_mask_path = os.path.join(args.masks_path, os.path.basename(mask_filename))
            target_mask_path = os.path.join(args.mask_output_dir, mask_new_name)
            shutil.copy(original_mask_path, target_mask_path)


print("图像生成完毕，已保存到文件夹中。")
