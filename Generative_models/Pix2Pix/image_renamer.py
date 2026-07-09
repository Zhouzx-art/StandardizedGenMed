import os
import argparse
import shutil
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_folder', type=str, default="None",
                        help='Source folder containing the generated fake_B images and their corresponding real_A images.')
    parser.add_argument('--img_folder', type=str, default="None",
                        help='Destination folder for renamed fake_B images (output generated images).')
    parser.add_argument('--mask_folder', type=str, default="None",
                        help='Destination folder for renamed real_A images (masks).')
    parser.add_argument('--start_num', type=int, default=1,
                        help='Starting index number for the output filename sequence (e.g., 1 -> 0001).')
    parser.add_argument('--dataset', type=str, default="None",
                        help='Dataset name prefix used in the output filenames (e.g., "ISIC" -> ISIC_0001_0000.png).')
    return parser.parse_args()


def process_images(src_folder, folder1, folder2,start_num,dataset):
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)

 
    all_files = os.listdir(src_folder)

    # 找出所有 fake_B 图片，并按字典序排序
    fake_images = sorted([f for f in all_files if 'fake_B' in f and f.lower().endswith('.png')])

    for idx, fake_name in enumerate(fake_images):
        # 构造编号
        dataset_id = f"{dataset}_{idx + start_num:04d}"

        # ---- (1) 复制 fake_B 并重命名 ----
        src_fake_path = os.path.join(src_folder, fake_name)
        dst_fake_name = f"{dataset_id}_0000.png"
        dst_fake_path = os.path.join(folder1, dst_fake_name)
        shutil.copy(src_fake_path, dst_fake_path)

        # ---- (2) 找到对应 real_A 并复制 ----
        real_name = fake_name.replace("fake_B", "real_A")
        src_real_path = os.path.join(src_folder, real_name)

        if os.path.exists(src_real_path):
            dst_real_name = f"{dataset_id}.png"
            dst_real_path = os.path.join(folder2, dst_real_name)
            shutil.copy(src_real_path, dst_real_path)
        else:
            print(f"对应 real_A 不存在: {real_name}")

    print(f"已处理 {len(fake_images)} 张 fake_B 图像，并复制对应 real_A 图像。")

args = get_args()

src_folder = args.src_folder
folder1 =args.img_folder
folder2 =args.mask_folder

process_images(src_folder, folder1, folder2,args.start_num,args.dataset)
