"""
Main function for arguments parsing
Author: Tal Daniel
"""
# imports
import torch
import argparse
from train_soft_intro_vae import train_soft_intro_vae
from train_soft_intro_vae import vae_generate
from train_soft_intro_vae import vae_generate_batch
import time

if __name__ == "__main__":
    """
        Recommended hyper-parameters:
        - CIFAR10: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
        - SVHN: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
        - MNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
        - FashionMNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
        - Monsters: beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, z_dim: 128, batch_size: 16
        - CelebA-HQ: beta_kl: 1.0, beta_rec: 0.5, beta_neg: 1024, z_dim: 256, batch_size: 8
    """
    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-d", "--dataset", type=str,
                        help="dataset to train on: ['cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', "
                             "'celeb256', 'celeb1024']")
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=250)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=128)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=2e-4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("-v", "--num_vae", type=int, help="number of epochs for vanilla vae training", default=0)
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss",
                        default=1.0)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence",
                        default=1.0)
    parser.add_argument("-e", "--beta_neg", type=float,
                        help="beta coefficient for the kl divergence in the expELBO function", default=1.0)
    parser.add_argument("-g", "--gamma_r", type=float,
                        help="coefficient for the reconstruction loss for fake data in the decoder", default=1e-8)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=-1)
    parser.add_argument("-p", "--pretrained", type=str, help="path to pretrained model, to continue training or generate images",
                        default="None")
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device",
                        default=-1)
    parser.add_argument('-f', "--fid", action='store_true', help="if specified, FID wil be calculated during training")
    ##
    parser.add_argument('--option', type=int,default=0)
    ##

    parser.add_argument("-t", "--test_name",type=str, default="test01")
    parser.add_argument("--save_dir",type=str, help="Save path of generated images", default="none")
    parser.add_argument('--start_num', type=int,default=0)
    parser.add_argument('--num_size', type=int,default=0)
    args = parser.parse_args()

    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))
    pretrained = None if args.pretrained == "None" else args.pretrained
    ##修改了save_interval
    if args.option==0 :
        start_time_total = time.time()
        train_soft_intro_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_workers=0,
                            num_epochs=args.num_epochs,
                            num_vae=args.num_vae, beta_kl=args.beta_kl, beta_neg=args.beta_neg, beta_rec=args.beta_rec,
                            device=device, save_interval=100, start_epoch=0, lr_e=args.lr, lr_d=args.lr,
                            pretrained=pretrained, seed=args.seed,
                            test_iter=1000, with_fid=args.fid,test_name=args.test_name)
        end_time_total = time.time()
        elapsed = end_time_total - start_time_total
        log = f"totally spend: {elapsed:.2f} 秒\n"

        # 写入txt文件
        path_2="./run_time/total_train_time/total_test_"+args.test_name+".txt"
        with open(path_2, 'a') as f:
            f.write(log)

        print(log)
    #用于生成对比图片
    elif args.option==1:
        vae_generate(dataset='ISIC', z_dim=256, seed=2,
                     pretrained='./saves/ISIC_soft_intro_betas_0.5_1024.0_1.0_model_epoch_249_iter_18500.pth',
                     batch_size=5, save_dir='./generated_images')
    #用于批量化生成图片
    else :
        vae_generate_batch(dataset=args.dataset, z_dim=256, seed=args.seed,device=device,
                     pretrained=args.pretrained,
                     batch_size=args.batch_size, save_dir=args.save_dir,start_num=args.start_num,start_fold="train",num_size=args.num_size)
        
        #start_num=585,start_fold="test",num_size=65
        #start_num=0,start_fold="train",num_size=585
