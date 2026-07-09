import os
from PIL import Image
import argparse
parser = argparse.ArgumentParser(description='Dataset preprocessing')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
args = parser.parse_args()

folder=['train','test']
for name in folder:
    imgs_dir = '/path/to/your/DATASET/'+args.dataset+'/imgs/'+name
    masks_dir = '/path/to/your/DATASET/'+args.dataset+'/masks/all/'+name

    # 输出文件夹
    output_dir = './datasets/'+args.dataset+'/'+name
    os.makedirs(output_dir, exist_ok=True)

    # 遍历 imgs 文件夹中的所有文件
    for filename in os.listdir(imgs_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(imgs_dir, filename)
            mask_path = os.path.join(masks_dir, filename)

            # 检查 masks 中是否存在同名文件
            if os.path.exists(mask_path):
                # 打开灰度图像
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('RGB')

                # 检查大小
                if img.size != (256, 256) or mask.size != (256, 256):
                    print(f"跳过大小不一致的文件：{filename}")
                    continue

                # 拼接图像（水平）
                new_img = Image.new('RGB', (512, 256))
                new_img.paste(img, (0, 0))
                new_img.paste(mask, (256, 0))

                # 保存拼接后的图像
                new_img.save(os.path.join(output_dir, filename))
            else:
                print(f"未找到对应的 mask 文件：{filename}")
