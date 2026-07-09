dataset=chaos
diffusion_img_size=256
diffusion_depth_size=1
diffusion_num_channels=3
batch_size=1
test_txt_dir=/home//path/to/your//A_DATASET/test.txt
dataset_root_dir=/home//path/to/your//A_DATASET/Chaos_resize/imgs/train
train_num_steps=1000001
cond_dim=48
results_folder=LeFusion/LeFusion_Model/Chaos

python LeFusion/train/train.py \
    dataset=$dataset \
    model.diffusion_img_size=$diffusion_img_size \
    model.diffusion_depth_size=$diffusion_depth_size \
    model.diffusion_num_channels=$diffusion_num_channels \
    dataset.test_txt_dir=$test_txt_dir \
    dataset.root_dir=$dataset_root_dir \
    model.train_num_steps=$train_num_steps \
    model.batch_size=$batch_size \
    model.cond_dim=$cond_dim \
    model.results_folder=$results_folder \

