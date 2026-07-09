import os
from PIL import Image
import argparse
def resize_and_save(hint_file, sample_file, labels_folder, images_folder, target_size, prefix, idx):
    # 打开掩码图像并缩放
    hint_img = Image.open(hint_file)
    hint_resized = hint_img.resize(target_size, Image.NEAREST)
    
    # 打开医学图像并缩放
    sample_img = Image.open(sample_file)
    sample_resized = sample_img.resize(target_size, Image.BILINEAR)
    
    # 保存缩放后的掩码图像
    label_output_name = f"{prefix}_{idx:04d}.png"
    label_output_path = os.path.join(labels_folder, label_output_name)
    hint_resized.save(label_output_path)
    
    # 保存缩放后的医学图像
    image_output_name = f"{prefix}_{idx:04d}_0000.png"
    image_output_path = os.path.join(images_folder, image_output_name)
    sample_resized.save(image_output_path)
    
    print(f"保存掩码图像: {label_output_path}")
    print(f"保存医学图像: {image_output_path}")

def process_images(hint_folder, sample_folder, labels_folder, images_folder, name_prefix,start_id):
    # 设置目标尺寸
    target_size = (256, 256)

    # 确保输出文件夹存在
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # 获取 hint 文件夹中的文件名并排序
    hint_files = sorted(os.listdir(hint_folder))

    # 遍历并处理每一对掩码和医学图像
    for idx, hint_file_name in enumerate(hint_files):
        if hint_file_name.endswith(".png"):
            # 构建文件的完整路径
            hint_file_path = os.path.join(hint_folder, hint_file_name)
            
            # 从 sample 文件夹中获取对应的医学图像路径
            sample_file_path = os.path.join(sample_folder, hint_file_name)
            
            # 调用函数进行缩放和保存
            resize_and_save(hint_file_path, sample_file_path, labels_folder, images_folder, target_size, name_prefix, idx + 1 + start_id)

# 你的输入文件夹路径
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
name_prefix = args.name_prefix
hint_folder = args.save_path+'/hint'
sample_folder = args.save_path+'/sample'
# 输出文件夹路径
labels_folder = args.masks_out_dir
images_folder = args.imgs_out_dir


# 调用处理函数
process_images(hint_folder, sample_folder, labels_folder, images_folder, name_prefix,args.start_id)
