from fld.metrics.FID import FID
from fld.metrics.KID import KID
from fld.metrics.FLD import FLD
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="None", help='VGG/ResNet')
    parser.add_argument('--generate_name', type=str, default="None", help='VGG/ResNet')
    return parser.parse_args()

args = get_args()

train_folder="path/to/your/DATASET/"+args.dataset_name+"_resize/imgs/train"
test_folder="path/to/your/DATASET/"+args.dataset_name+"_resize/imgs/test"
generate_folder="./nnUNet/DATASET/nnUNet_raw/Dataset"+args.generate_name+"/imagesTr"
feature_extractor = InceptionFeatureExtractor()

imgs_train_feat = feature_extractor.get_dir_features(train_folder, extension="png")
imgs_test_feat = feature_extractor.get_dir_features(test_folder, extension="png")
gen_feat  = feature_extractor.get_dir_features(generate_folder, extension="png")
fid_val = FID().compute_metric(imgs_train_feat, None,gen_feat)
print(f"FID: {fid_val:.3f}")

# Like FID, can get either Train or Test KID
kid_val=KID().compute_metric(imgs_train_feat, None,gen_feat)

print(f"KID: {kid_val:.3f}")
fld_val = FLD().compute_metric(imgs_train_feat, imgs_test_feat, gen_feat)
print(f"FLD: {fld_val:.3f}")

with open("fid_kid_fld_score.txt", "a") as f:
    result_str = f"{args.generate_name}:\nFID: {fid_val:.3f}\nKID: {kid_val:.3f}\nFLD: {fld_val:.3f}\n\n"
    f.write(result_str)