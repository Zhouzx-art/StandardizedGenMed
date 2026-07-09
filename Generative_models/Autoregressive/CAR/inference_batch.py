import os
import argparse
import torch
import torchvision
import random
import numpy as np
import cv2
import PIL.Image as PImage
from models import VQVAE, build_car
from utils.loading_utils import load_image
from utils.control_data_utils import pil_to_numpy, numpy_to_pt
import shutil
import time
def main(args):
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

    MODEL_DEPTH = 16
    assert MODEL_DEPTH in {16, 20, 24, 30}

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae, car = build_car(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1, depth=MODEL_DEPTH, shared_aln=False,
    )

    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)

    var_weights = torch.load(args.var_ckpt, map_location='cpu')
    car_weights = torch.load(args.car_ckpt, map_location='cpu')
    all_weights = {}
    all_weights.update(var_weights)
    all_weights.update(car_weights)
    print("=> Loading CAR weights...")
    car.load_state_dict(all_weights, strict=True)
    total_params = sum(p.numel() for p in car.parameters())
    print(f"Total number of parameters in car_ckpt: {total_params:,}")
    print("=> CAR weights loaded.")
    vae.eval(), car.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in car.parameters():
        p.requires_grad_(False)

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    def get_control_for_each_scale(control_image, scale):
        def normalize_01_into_pm1(x):
            return x.add(x).add_(-1)
        c_tensors = []
        c_images = []
        for pn in scale:
            c_res = control_image.resize((pn * 16, pn * 16))
            c_images.append(c_res)
            c_tensors.append(normalize_01_into_pm1(numpy_to_pt(pil_to_numpy(c_res))))
        return c_images, c_tensors

    # 遍历 mask 文件夹
    if not os.path.exists(args.img_output_dir):
        os.makedirs(args.img_output_dir, exist_ok=True)
    mask_files = sorted([f for f in os.listdir(args.mask_dir) if f.endswith('.png')])
    cnt=0
    start_time=0
    for mask_file in mask_files:
        cnt=cnt+1
        if cnt%16==1:
            start_time=time.time()
        mask_path = os.path.join(args.mask_dir, mask_file)
        new_name = f"{args.dataset_name}_{cnt+args.start_num:04d}_0000.png"
        mask_new_name=f"{args.dataset_name}_{cnt+args.start_num:04d}.png"
        # 加载灰度 mask，并转为 RGB
        mask = PImage.open(mask_path).convert('L')
        mask_np = np.array(mask)
        mask_np_rgb = np.stack([mask_np] * 3, axis=-1)
        control = PImage.fromarray(mask_np_rgb.astype(np.uint8))

        control_images, control_tensors = get_control_for_each_scale(control, patch_nums)
        class_labels = [1] * 2
        B = len(class_labels)
        label_B = torch.tensor(class_labels, device=device)

        for i in range(len(control_tensors)):
            control_tensors[i] = control_tensors[i].repeat(B, 1, 1, 1).to(device)

        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                recon_B3HW = car.car_inference(B=B, label_B=label_B, cfg=0,
                                               top_k=900, top_p=0.95, g_seed=seed,
                                               more_smooth=False, control_tensors=control_tensors)

        chw = torchvision.utils.make_grid(recon_B3HW[1], nrow=1, padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))

       
        combined = PImage.new('RGB', (chw.width, chw.width))
        combined.paste(chw, (0, 0))
        if cnt%16==0:
            end_time=time.time()-start_time
            print("生产16张图片耗时:",end_time)
        save_path = os.path.join(args.img_output_dir, new_name)
        combined.save(save_path)
        if not os.path.exists(args.mask_output_dir):
            os.makedirs(args.mask_output_dir)
        target_mask_path = os.path.join(args.mask_output_dir, mask_new_name)
        shutil.copy(mask_path, target_mask_path)
        print(f"[✓] Saved: {save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAR Inference on Mask Folder')
    parser.add_argument('--dataset_name', type=str, default='Chaos')
    parser.add_argument('--vae_ckpt', type=str, default='path/to/CAR/pre_ckpt/vae_ckpt/vae_ch160v4096z32.pth')
    parser.add_argument('--var_ckpt', type=str, default='path/to/CAR/pre_ckpt/var_ckpt/var_d16.pth')
    parser.add_argument('--car_ckpt', type=str, default='path/to/CAR/local_output_hed_new_l5/car_ckpt_last.pth')
    parser.add_argument('--mask_dir', type=str, default='path/to/your/DATASET/Chaos_resize/masks/all/train')
    parser.add_argument('--img_output_dir', type=str, default='path/to/nnUNet/DATASET/')
    parser.add_argument('--mask_output_dir', type=str, default='path/to/nnUNet/DATASET/')
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    main(args)
