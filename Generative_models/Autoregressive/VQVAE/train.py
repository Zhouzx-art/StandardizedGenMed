import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from torchvision.utils import save_image
from tqdm import tqdm
import os
import time
import lpips
from vqvae import VQVAE  # 你的多尺度 VQ-VAE 模型定义
from torch.utils.data import Dataset
from PIL import Image
import argparse

class MyImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) 
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # 标签填 0 占位


class MyImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) 
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # 标签填 0 占位
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Name of the dataset, used for naming checkpoint and reconstruction save directories.')
    parser.add_argument('--data_path', type=str,help='Path to the directory containing training images.')
    parser.add_argument('--batch_size', type=int, default=8,help='Batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Batch size for validation.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4,help='Learning rate for the Adam optimizer.')
    parser.add_argument('--save_iter', type=int, default=50,help='Save model checkpoint every N epochs.')
    parser.add_argument('--resume_path', type=str, default="./vae_ckpt/vae_ch160v4096z32.pth",help='Pre-trained weights of VAE.')
    return parser.parse_args()
# ===== 多 GPU 环境设置 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  
args = get_args()


# ===== 路径和超参数设置 =====
data_path = args.data_path
save_path = 'checkpoints_'+str(args.dataset_name)+'/'
img_save_path = 'recon_vis_'+str(args.dataset_name)+'/'
time_path='./time/total_'+str(args.dataset_name)+'.txt'

epochs = args.epochs
batch_size = args.batch_size
learning_rate =args.lr
lambda_P = 0.5  # 感知损失权重
img_size = 256

# ===== 创建文件夹 =====
os.makedirs(save_path, exist_ok=True)
os.makedirs(img_save_path, exist_ok=True)
os.makedirs('./time/', exist_ok=True)

# ===== 图像预处理 =====
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # [0,1] -> [-1,1]
])

# ===== 数据加载 =====
#train_dataset = datasets.ImageFolder(data_path, transform=transform)
train_dataset = MyImageDataset(data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ===== 模型、优化器和损失函数初始化 =====
model = VQVAE(test_mode=False).to(device)
model = nn.DataParallel(model)  # 多 GPU 包装

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

# 感知损失
perceptual_loss = lpips.LPIPS(net='vgg').to(device)
perceptual_loss.eval()
perceptual_loss = nn.DataParallel(perceptual_loss)  # 多 GPU 包装

# ===== 加载预训练模型（可选） =====
resume_path =args.resume_path
start_epoch = 0
if os.path.exists(resume_path):
    print(f"[加载模型] 从 {resume_path} 加载继续训练...")
    checkpoint = torch.load(resume_path, map_location=device)
    model.module.load_state_dict(checkpoint)  # 注意 .module

# ===== 可视化重建函数 =====
def visualize_reconstruction(model, loader, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x = x.to(device)
        recon = model.module.img_to_reconstructed_img(x, last_one=True)  # 注意 .module

        # [-1,1] -> [0,1]
        x_vis = x * 0.5 + 0.5
        recon_vis = recon * 0.5 + 0.5

        grid = utils.make_grid(torch.cat([x_vis, recon_vis], dim=0), nrow=batch_size)
        save_file = os.path.join(save_dir, f'epoch_{epoch:03d}.png')
        utils.save_image(grid, save_file)
        print(f'[可视化保存] {save_file}')

# ======= 训练循环 =======
start_time = time.time()
for epoch in range(start_epoch, epochs):
    start_time_epoch = time.time()
    model.train()
    running_loss = 0.0

    for batch_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device)

        optimizer.zero_grad()
        recon_imgs, _, vq_loss = model(imgs)  # 多 GPU 自动处理
        recon_imgs = recon_imgs.clamp_(-1, 1)

        l2_image = mse_loss(recon_imgs, imgs)
        lp = perceptual_loss(recon_imgs, imgs).mean()

        loss = l2_image + lambda_P * lp + vq_loss
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[Epoch {epoch+1}/{epochs}] Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        if (epoch + 1) % 20 == 0 or epoch == 0:
            visualize_reconstruction(model, train_loader, epoch, img_save_path)

        if (epoch + 1) % args.save_iter == 0 or epoch == 0:
            torch.save(model.module.state_dict(), os.path.join(save_path, f'vqvae_epoch{epoch+1}.pth'))

    end_time_epoch = time.time() - start_time_epoch
    log_epoch = f"[Epoch {epoch+1}] totally spend: {end_time_epoch:.2f} 秒\n"
    with open(time_path, 'a') as f:
        f.write(log_epoch)

end_time = time.time() - start_time
log = f"[All Epochs] totally spend: {end_time:.2f} 秒\n"
with open(time_path, 'a') as f:
    f.write(log)
