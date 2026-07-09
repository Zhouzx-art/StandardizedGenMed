import os
from PIL import Image
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Normalize image filenames")
    
    parser.add_argument("--imgs_out_dir", type=str, default="",
                        help="Output directory for normalized images")
    parser.add_argument("--masks_out_dir", type=str, default="",
                        help="Output directory for normalized masks")
    parser.add_argument("--dataset_name", type=str, default="FeTS",
                        help="Name of the dataset")
    parser.add_argument("--imgs_dir", type=str,
                        default="./ddpm-FeTS_1-256-segguided/samples_many_epoch_599_24256_1",
                        help="Input directory containing generated images")
    parser.add_argument("--masks_dir", type=str,
                        default="/path/to/your/DATASET/FeTS_resize/masks/all/train",
                        help="Input directory containing original masks")
    parser.add_argument("--start_num", type=int, default=0,
                        help="Starting index number for renaming")
    
    return parser.parse_args()

args = parse_args()
imgs_out_dir = args.imgs_out_dir
masks_out_dir = args.masks_out_dir
dataset_name = args.dataset_name
imgs_dir = args.imgs_dir
masks_dir = args.masks_dir
start_num = args.start_num

# 创建输出文件夹（如果不存在）
os.makedirs(imgs_out_dir, exist_ok=True)
os.makedirs(masks_out_dir, exist_ok=True)

# 获取并排序imgs中的文件
img_files = sorted([f for f in os.listdir(imgs_dir) if f.endswith('.png')])

# 遍历排序后的文件
for idx, img_name in enumerate(img_files, start=1):
    # 打开图像
    img_path = os.path.join(imgs_dir, img_name)
    img = Image.open(img_path)

     # 构造输出文件名
    new_img_name = f'{dataset_name}_{start_num+idx:04d}_0000.png'
    new_img_path = os.path.join(imgs_out_dir, new_img_name)
    img.save(new_img_path)
    
    # 从原始图像名中去掉"condon_"
    mask_name = img_name.replace("condon_", "")
    mask_path = os.path.join(masks_dir, mask_name)

    # 检查 mask 是否存在
    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
        new_mask_name = f'{dataset_name}_{start_num+idx:04d}.png'
        new_mask_path = os.path.join(masks_out_dir, new_mask_name)
        mask.save(new_mask_path)
    else:
        print(f"[警告] 找不到 mask：{mask_name}")

