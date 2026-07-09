import os
import io
import blobfile as bf
import torch as th
import json
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm import Unet3D, GaussianDiffusion_Nolatent
from get_dataset.get_dataset import get_inference_dataloader
import torchio as tio
import yaml
from omegaconf import DictConfig
import hydra
from PIL import Image
def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


def perturb_tensor(tensor, mean=0.0, std=1.0, bili=0.1):
    perturbation = th.normal(mean, std, size=tensor.size())
    perturbation -= perturbation.mean()
    max_perturbation = tensor.abs() * bili
    perturbation = perturbation / perturbation.abs().max() * max_perturbation
    perturbed_tensor = tensor + perturbation
    return perturbed_tensor


@hydra.main(config_path='confs', config_name='infer', version_base=None)
def main(conf: DictConfig):
    data_type = conf['data_type'].lower()
    if data_type not in ['lidc', 'emidec','isic','polyp','prostate','nuclei','fundus','fets','chaos']:
        raise ValueError("Wrong data type")
    print("Start", data_type)
    device = dev(conf.get('device'))

    model = Unet3D(
        dim=conf.diffusion_img_size,
        dim_mults=conf.dim_mults,
        channels=conf.diffusion_num_channels,
        cond_dim=conf.cond_dim,
    )

    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=conf.diffusion_img_size,
        num_frames=conf.diffusion_depth_size,
        channels=conf.diffusion_num_channels,
        timesteps=conf.timesteps,
        loss_type=conf.loss_type,
        data_type=data_type,
    )
    diffusion.to(device)

    weights_dict = {}
    for k, v in (load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")["model"].items()):
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    diffusion.load_state_dict(weights_dict)

    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Params: {total_params / 1e6:.2f} M")
    show_progress = conf.show_progress

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'hist_clusters', f'{data_type}_clusters.json')
    with open(file_path, 'r') as f:
        clusters = json.load(f)

    cluster_centers = clusters[0]['centers']

    print("sampling...")

    dl = get_inference_dataloader(dataset_root_dir=conf.dataset_root_dir, test_txt_dir=conf.test_txt_dir, batch_size=conf.batch_size, data_type=data_type)  
    
    idx = 0
    for batch in iter(dl):
        for type in range(conf.types):
            print("idx:",idx+1)
            print("type_of_cond:", type+1)
            if data_type == 'lidc':
                hist = th.tensor(cluster_centers[type])
                hist = perturb_tensor(tensor=hist)
                hist = hist.unsqueeze(0)
            elif data_type == 'isic' or data_type == 'polyp' or data_type == 'prostate' or data_type == 'nuclei' or data_type == 'fets' or data_type == 'chaos':
                hist = th.tensor(cluster_centers[type])
                hist = perturb_tensor(tensor=hist)
                hist = hist.unsqueeze(0)
            elif data_type == 'emidec':
                hist_1 = perturb_tensor(th.tensor(cluster_centers[0]))
                hist_2 = perturb_tensor(th.tensor(cluster_centers[1]))
                hist = th.cat((hist_1, hist_2), dim=0).to(device)
            elif data_type == 'fundus':
                hist_1 = perturb_tensor(th.tensor(cluster_centers[0]))
                hist_2 = perturb_tensor(th.tensor(cluster_centers[1]))
                hist = th.cat((hist_1, hist_2), dim=0).to(device)
            for k in batch.keys():
                if isinstance(batch[k], th.Tensor):
                    batch[k] = batch[k].to(device)
            model_kwargs = {}
            model_kwargs["gt"] = batch['GT']
            gt_keep_mask = batch.get('gt_keep_mask')
            if gt_keep_mask is not None:
                model_kwargs['gt_keep_mask'] = gt_keep_mask
            batch_size = model_kwargs["gt"].shape[0]

            sample_fn = diffusion.p_sample_loop_repaint

            print(hist.shape)
            output = sample_fn(
                shape = (batch_size, conf.diffusion_num_channels, conf.diffusion_depth_size, conf.diffusion_img_size, conf.diffusion_img_size),
                model_kwargs=model_kwargs,
                device=device,
                progress=show_progress,
                conf=conf,
                cond=hist
            )

            if data_type == 'lidc':
                image_fold = f"Image_{type+1}"
                label_fold = f"Mask_{type+1}"
                os.makedirs(os.path.join(conf.target_img_path, image_fold), exist_ok=True)
                os.makedirs(os.path.join(conf.target_label_path, label_fold), exist_ok=True)
                for i in range(batch_size):
                    result = output[i, :, :, :, :].cpu()
                    restore_affine = batch['affine'][i].squeeze(0).cpu()
                    gt_name = batch['GT_name'][i]
                    name_part, extension = gt_name.rsplit('.nii.gz', 1)[0], '.nii.gz'
                    main_name, vol_part = name_part.rsplit('_CVol_', 1)
                    mask_name = f"{main_name}_Mask_{vol_part}{extension}"
                    gen_image = tio.ScalarImage(tensor=result, channels_last=False, affine=restore_affine)
                    gen_image.save(os.path.join(conf.target_img_path, image_fold, gt_name))
                    label = batch['gt_keep_mask'][i].cpu()
                    label = tio.LabelMap(tensor=label, channels_last=False, affine=restore_affine)
                    label.save(os.path.join(conf.target_label_path, label_fold, mask_name))
            elif data_type == 'isic' or data_type == 'polyp' or data_type == 'prostate' or data_type == 'nuclei' or data_type == 'fets' or data_type == 'chaos':
                image_fold = f"Image_{type+1}"
                label_fold = f"Mask_{type+1}"
                os.makedirs(os.path.join(conf.target_img_path, image_fold), exist_ok=True)
                os.makedirs(os.path.join(conf.target_label_path, label_fold), exist_ok=True)
                for i in range(batch_size):
                    # 获取生成的图像数据
                    result = output[i, :, :, :, :].cpu()
                    print(result.shape)
                    result_img = result[:, 0, :, :].cpu()
                    print(result_img.shape)
                    # 由于生成的是 RGB 图像，我们将其转换为 [H, W, C] 格式，并缩放至[0, 255]
                    result_img = (result_img * 0.5) + 0.5  # 还原到 [0, 1] 范围
                    result_img = result_img.permute(1, 2, 0)  # 从 [C, H, W] 转为 [H, W, C]
                    result_img = (result_img * 255).clamp(0, 255).byte()  # 缩放到 [0, 255] 并转换为 byte
                    
                    # 将图像转换为PIL图像（RGB）
                    result_img_pil = Image.fromarray(result_img.numpy(), mode='RGB')
                    
                    # 获取GT名称，假设它是 ISIC2017 数据集中图像的文件名
                    gt_name = batch['GT_name'][i]  # 原图像的名称
                    name_part, extension = gt_name.rsplit('.png', 1)
                    mask_name = f"{name_part}_mask.png"  # 假设掩码是 .png 格式
                    
                    # 将生成的图像保存为 RGB 格式的 PNG 文件
                    result_img_pil.save(os.path.join(conf.target_img_path, image_fold, gt_name))  # 保存图像
                    
                    # 获取并保存掩码
                    mask = batch['gt_keep_mask'][i].cpu()  # 获取掩码
                    mask = mask.squeeze(0).byte()  # 移除多余的维度并转换为 byte（灰度图）
                    
                    # 确保掩码是二维数组
                    if mask.ndimension() > 2:
                        mask = mask[0, :, :]  # 取第一个通道，确保是 [H, W]
                    
                    # 将掩码转换为PIL图像（灰度模式）
                    mask_pil = Image.fromarray(mask.numpy(), mode='L')  # 'L' 模式为灰度图
                    
                    # 保存掩码为 PNG 文件
                    mask_pil.save(os.path.join(conf.target_label_path, label_fold, mask_name))  # 保存掩码
            elif data_type == 'fundus':
                #output = output.permute(0, 1, 3, 4, 2).cpu()
                image_fold = f"Image_{type+1}"
                label_fold = f"Mask_{type+1}"
                os.makedirs(os.path.join(conf.target_img_path, image_fold), exist_ok=True)
                os.makedirs(os.path.join(conf.target_label_path, label_fold), exist_ok=True)
                for i in range(batch_size):
                    result = output[i, :, :, :, :].cpu()
                    print(result.shape)
                    result_img = result[:, 0, :, :].cpu()
                    print(result_img.shape)
                    result_img = (result_img * 0.5) + 0.5  # 还原到 [0, 1] 范围
                    result_img = result_img.permute(1, 2, 0)  # 从 [C, H, W] 转为 [H, W, C]
                    result_img = (result_img * 255).clamp(0, 255).byte()  # 缩放到 [0, 255] 并转换为 byte
                     # 将图像转换为PIL图像（RGB）
                    result_img_pil = Image.fromarray(result_img.numpy(), mode='RGB')
                    
                    # 获取GT名称，假设它是 ISIC2017 数据集中图像的文件名
                    gt_name = batch['GT_name'][i]  # 原图像的名称
                    name_part, extension = gt_name.rsplit('.png', 1)
                    mask_name = f"{name_part}_mask.png"  # 假设掩码是 .png 格式
                    
                    # 将生成的图像保存为 RGB 格式的 PNG 文件
                    result_img_pil.save(os.path.join(conf.target_img_path, image_fold, gt_name))  # 保存图像
                    
                    # 获取并保存掩码
                    mask = batch['gt_keep_mask'][i].cpu()  # 获取掩码
                    mask = mask.squeeze(0).byte()  # 移除多余的维度并转换为 byte（灰度图）
                    
                    # 确保掩码是二维数组
                    if mask.ndimension() > 2:
                        mask = mask[0, :, :]  # 取第一个通道，确保是 [H, W]
                    
                    # 将掩码转换为PIL图像（灰度模式）
                    mask_pil = Image.fromarray(mask.numpy(), mode='L')  # 'L' 模式为灰度图
                    
                    # 保存掩码为 PNG 文件
                    mask_pil.save(os.path.join(conf.target_label_path, label_fold, mask_name))  # 保存掩码
            elif data_type == 'emidec':
                output = output.permute(0, 1, 3, 4, 2).cpu()
                os.makedirs(conf.target_img_path, exist_ok=True)
                os.makedirs(conf.target_label_path, exist_ok=True)
                for i in range(batch_size):
                    result = output[i, :, :, :, :].cpu()
                    restore_affine = batch['affine'][i].squeeze(0).cpu()
                    gt_name = batch['GT_name'][i]
                    gen_image = tio.ScalarImage(tensor=result, channels_last=False, affine=restore_affine)
                    gen_image.save(os.path.join(conf.target_img_path, gt_name))
                    label = batch['gt_keep_mask'][i].cpu()
                    label = tio.LabelMap(tensor=label, channels_last=False, affine=restore_affine)
                    label.save(os.path.join(conf.target_label_path, gt_name))


        idx += 1

    print("sampling complete")


if __name__ == "__main__":
    main()