data_type=fundus
types=1
diffusion_img_size=256
diffusion_depth_siz=1
diffusion_num_channels=6
batch_size=16
dataset_root_dir=/path/to/your/DATASET/Fundus_resize/imgs/train
target_img_path=data/FUNDUS/gen/Image/
target_label_path=data/FUNDUS/gen/Mask/
model_path=LeFusion/LeFusion_Model/FUNDUS/model-2.pt
jump_length=2
jump_n_sample=2
cond_dim=96

python LeFusion/inference/inference.py \
    data_type=$data_type \
    types=$types\
    diffusion_img_size=$diffusion_img_size \
    diffusion_depth_size=$diffusion_depth_siz \
    diffusion_num_channels=$diffusion_num_channels \
    dataset_root_dir=$dataset_root_dir \
    target_img_path=$target_img_path \
    target_label_path=$target_label_path \
    schedule_jump_params.jump_length=$jump_length \
    schedule_jump_params.jump_n_sample=$jump_n_sample \
    model_path=$model_path \
    cond_dim=$cond_dim \
    batch_size=$batch_size



