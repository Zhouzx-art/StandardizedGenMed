# implementation of Rectified Flow with AMP
import argparse
import torch
import os

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")


class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, mask):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, mask)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, mask, sample_steps=50, null_cond=None, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        for s in range(sample_steps):
            t_val = 1.0 - s / sample_steps
            t = torch.tensor([t_val] * b).to(z.device)

            vtheta = self.model(z, t, mask)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vtheta = vu + cfg * (vtheta - vu)

            z = z - dt * vtheta
            images.append(z)

        return images


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="None",
                        help='Name of the dataset')
    parser.add_argument('--masks_path', type=str, default="None",
                        help='Path to the directory containing corresponding masks for the images.')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=4,
                        help='Batch size for validation during training.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Total number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the Adam optimizer.')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Patch size for the DiT model.')
    parser.add_argument('--save_iter', type=int, default=100,
                        help='Save model checkpoint every N epochs.')
    return parser.parse_args()


if __name__ == "__main__":
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from tqdm import tqdm
    from dit import DiT_Llama
    from dataset import ORIGADataset, VALDataset
    from torchvision import transforms
    import time

    start_time_total = time.time()
    args = get_args()

    input_size = 256
    channels = 3
    val_batch_size = args.val_batch_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr

    model = DiT_Llama(
        in_channels=channels,
        input_size=input_size,
        patch_size=args.patch_size,
        dim=1024,
        n_layers=10,
        n_heads=8,
    )
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    scaler = torch.cuda.amp.GradScaler()

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    mask_transform_in = transforms.Compose([
        transforms.Resize((32, 32), Image.NEAREST),
        transforms.ToTensor(),
    ])
    imgs_path=args.masks_path.replace("masks/all", "imgs")
    train_dataset = ORIGADataset(images_dir=imgs_path, masks_dir=args.masks_path,
                                  transform=image_transform, mask_transform=mask_transform_in)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    for epoch in range(num_epochs):
        start_time_total_epoch = time.time()
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}

        for i, (x, mask) in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}"):
            x, mask = x.cuda(), mask.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, blsct = rf.forward(x, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            path_loss_test = f"./loss/training_loss_{args.dataset_name}.txt"
            with open(path_loss_test, "a") as f:
                f.write(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}\n")

            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        epoch_loss = sum(lossbin.values()) / sum(losscnt.values())
        with open(path_loss_test, "a") as f:
            f.write(f"Epoch {epoch} finished, Average Loss: {epoch_loss}\n")
        path_loss_bin_test = f"./loss/training_loss_bin_{args.dataset_name}.txt"
        with open(path_loss_bin_test, "a") as f:
            for i in range(10):
                print(f"[Epoch {epoch}] Loss bin {i}: {lossbin[i] / losscnt[i]:.4f}")
                f.write(f"[Epoch {epoch}] Loss bin {i}: {lossbin[i] / losscnt[i]:.4f}\n")

        rf.model.eval()
        start_time_generate = time.time()
        with torch.no_grad():
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1 or epoch == 0:
                val_dataset = VALDataset(
                    images_dir=imgs_path,
                    masks_dir=args.masks_path,
                    transform=image_transform,
                    mask_transform=mask_transform_in
                )
                val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
                val_x, val_mask = next(iter(val_loader))
                val_x, val_mask = val_x.cuda(), val_mask.cuda()

                init_noise = torch.randn(val_batch_size, channels, input_size, input_size).cuda()
                null_cond = torch.zeros_like(val_mask).cuda()
                fake_images = rf.sample(init_noise, val_mask, sample_steps=50, null_cond=None)[-1]

                def normalize_img(t):
                    return t * 0.5 + 0.5

                real_images = normalize_img(val_x).clamp(0, 1).cpu()
                fake_images = normalize_img(fake_images).clamp(0, 1).cpu()

                concat = torch.cat([real_images, fake_images], dim=0)
                grid = make_grid(concat, nrow=4)
                img = grid.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)

                folder_path = f"contents_{args.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                Image.fromarray(img).save(f"{folder_path}/sample_{epoch}_compare_2.png")

        if (epoch + 1) % args.save_iter == 0 or epoch == num_epochs - 1:
            checkpoint_dir = f"checkpoints_{args.dataset_name}"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save({
                'model': model.module.state_dict(),
                'scaler': scaler.state_dict()
            }, f"{checkpoint_dir}/dit_{epoch}.pt")

        end_time_generate = time.time()
        rf.model.train()
        end_time_total_epoch = time.time()
        elapsed_epoch = end_time_total_epoch - start_time_total_epoch - (end_time_generate - start_time_generate)
        log_epoch = f"epoch {epoch} spend: {elapsed_epoch:.2f} \u79d2\n"
        path_1 = f"./run_time/epoch_train_time/epoch_{args.dataset_name}.txt"
        with open(path_1, 'a') as f:
            f.write(log_epoch)

    end_time_total = time.time()
    elapsed = end_time_total - start_time_total
    log = f"totally spend: {elapsed:.2f} \u79d2\n"
    path_2 = f"./run_time/total_train_time/total_{args.dataset_name}.txt"
    with open(path_2, 'a') as f:
        f.write(log)
