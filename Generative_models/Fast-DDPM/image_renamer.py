import os
import shutil
import argparse
parser = argparse.ArgumentParser(description='Process medical images and corresponding masks.')
parser.add_argument('--name_prefix', type=str, required=True, help='Dataset name')
parser.add_argument('--save_path', type=str, required=True, help='Input directory containing generated images')
parser.add_argument('--start_id', type=int, required=True, help='Prefix for the output file names')   
parser.add_argument("--imgs_out_dir", type=str, default="",
                        help="Output directory for normalized images")
parser.add_argument("--masks_out_dir", type=str, default="",
                        help="Output directory for normalized masks")
# 解析命令行参数
args = parser.parse_args()

# Target directories
target_image_dir = args.imgs_out_dir
target_mask_dir = args.masks_out_dir

# Path to masks folder
mask_dir = f"/path/to/your/DATASET/{args.name_prefix}_resize/masks/all/train"

# Ensure the target directories exist
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_mask_dir, exist_ok=True)

# Process the generated images
image_counter = args.start_id
for img_name in os.listdir(args.save_path):
    # 跳过 _gt 文件
    if "_gt" in img_name:
        continue

    # 只处理 _pt 文件
    if "_pt" not in img_name:
        continue

    # 生成对应的 mask 文件名（去除 _pt 后缀）
    if img_name.endswith("_pt.png"):
        mask_name = img_name[:-7] + ".png"   # 去掉 "_pt.png" 保留 ".png"
    else:
         mask_name = img_name.replace("_pt", "")  # 备用处理方式

    # 当前序号
    cur_idx = image_counter

    # 新图像名称（nnUNet 格式）
    new_image_name = f"{args.name_prefix}_{cur_idx:04d}_0000.png"
    src_img_path = os.path.join(args.save_path, img_name)
    dst_img_path = os.path.join(target_image_dir, new_image_name)

    # 复制生成图像
    shutil.copy(src_img_path, dst_img_path)

    # 新 mask 名称
    new_mask_name = f"{args.name_prefix}_{cur_idx:04d}.png"
    src_mask_path = os.path.join(mask_dir, mask_name)
    dst_mask_path = os.path.join(target_mask_dir, new_mask_name)

    # 检查 mask 文件是否存在
    if os.path.exists(src_mask_path):
        shutil.copy(src_mask_path, dst_mask_path)
    else:
        print(f"警告：未找到对应的 mask 文件 {src_mask_path}，跳过该样本。")

    image_counter += 1

print("Processing completed!")