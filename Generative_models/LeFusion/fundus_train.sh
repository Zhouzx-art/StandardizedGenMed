dataset=fundus
diffusion_img_size=256
diffusion_depth_size=1
diffusion_num_channels=6
batch_size=1
dataset_root_dir=/path/to/your/DATASET/Fundus_resize/imgs/train
train_num_steps=1000001
cond_dim=96
results_folder=LeFusion/LeFusion_Model/FUNDUS

python LeFusion/train/train.py \
    dataset=$dataset \
    model.diffusion_img_size=$diffusion_img_size \
    model.diffusion_depth_size=$diffusion_depth_size \
    model.diffusion_num_channels=$diffusion_num_channels \
    model.batch_size=$batch_size \
    dataset.root_dir=$dataset_root_dir \
    model.train_num_steps=$train_num_steps \
    model.cond_dim=$cond_dim \
    model.results_folder=$results_folder \
    model.batch_size=$batch_size \

