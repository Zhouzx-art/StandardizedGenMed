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
    # Generate new image file name (e.g., Prostate_0001_0000.png)
    new_image_name = f"{args.name_prefix}_{image_counter:04d}_0000.png"
    image_counter += 1
        
    # Full path of the generated image
    src_img_path = os.path.join(args.save_path, img_name)
    dst_img_path = os.path.join(target_image_dir, new_image_name)
        
    # Copy and rename the generated image
    shutil.copy(src_img_path, dst_img_path)
        
    # Get corresponding mask image name (same as the generated image name)
    mask_name = img_name  # Same name as the original image
    src_mask_path = os.path.join(mask_dir, mask_name)
        
    # Generate new mask file name (e.g., Prostate_0001.png)
    new_mask_name = f"{args.name_prefix}_{image_counter-1:04d}.png"
    dst_mask_path = os.path.join(target_mask_dir, new_mask_name)
        
    # Copy and rename the mask image
    shutil.copy(src_mask_path, dst_mask_path)

print("Processing completed!")
