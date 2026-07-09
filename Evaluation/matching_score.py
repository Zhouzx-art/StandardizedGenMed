import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)


def multi_class_dice(y_true, y_pred, num_classes, smooth=1e-6):
    dice_scores = []
    for c in range(1, num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        intersection = np.sum(true_c & pred_c)
        union = np.sum(true_c) + np.sum(pred_c)
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    return np.mean(dice_scores)


def evaluate_dice(seg_dir, pre_dir, num_classes, name, way, filepath):
    seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.png')])
    pre_files = sorted([f for f in os.listdir(pre_dir) if f.endswith('.png')])

    assert len(seg_files) == len(pre_files), "Mismatch between ground truth and prediction file counts"

    dice_list = []

    for seg_name, pre_name in tqdm(zip(seg_files, pre_files), total=len(seg_files), desc="Evaluating"):
        assert seg_name == pre_name, f"Filename mismatch: {seg_name} vs {pre_name}"

        seg_path = os.path.join(seg_dir, seg_name)
        pre_path = os.path.join(pre_dir, pre_name)

        gt = np.array(Image.open(seg_path).convert('L'))
        pred = np.array(Image.open(pre_path).convert('L'))

        assert gt.shape == pred.shape, f"Shape mismatch for {seg_name}"

        if num_classes == 2:
            dice = dice_coefficient((gt > 0).astype(np.uint8),
                                    (pred > 0).astype(np.uint8))
        else:
            dice = multi_class_dice(gt, pred, num_classes)

        dice_list.append(dice)

        dice_path = f"./score_list/{name}_{way}.txt"

        base_name = pre_name.split('.')[0]
        ext = pre_name.split('.')[-1]
        new_name = f"{base_name}_0000.{ext}"

        with open(dice_path, 'a', encoding='utf-8') as f:
            f.write(new_name + ":\n")
            f.write(str(dice) + "\n")

    dice_array = np.array(dice_list)

    print(f"\nTotal evaluated images: {len(dice_list)}")
    print(f"Mean Dice score: {dice_array.mean():.4f}")
    print(f"Dice standard deviation: {dice_array.std():.4f}")

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"Method: {way}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total evaluated images: {len(dice_list)}\n")
        f.write(f"Mean Dice score: {dice_array.mean():.4f}\n")
        f.write(f"Dice standard deviation: {dice_array.std():.4f}\n")
        f.write("-" * 50 + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="Dice Evaluation for Segmentation")

    parser.add_argument("--name", type=str, default="Chaos", help="Dataset name")
    parser.add_argument("--way", type=str, default="SegGuidedDif", help="Method name")
    parser.add_argument("--filepath", type=str, default="score.txt", help="Output summary file path")
    parser.add_argument("--seg_dir", type=str, default="path/to/your/labelsTr", help="Ground truth directory")
    parser.add_argument("--pre_dir", type=str, default="path/to/your/predicted/labelsTr", help="Prediction directory")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of segmentation classes")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.name == "Fundus":
        args.num_classes = 3

    evaluate_dice(
        seg_dir=args.seg_dir,
        pre_dir=args.pre_dir,
        num_classes=args.num_classes,
        name=args.name,
        way=args.way,
        filepath=args.filepath
    )