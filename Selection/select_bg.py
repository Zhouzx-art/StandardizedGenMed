import os
from PIL import Image
import numpy as np
import argparse


# =========================
# Utility: load scores from txt
# =========================
def load_score(file_path):
    data = {}
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]

    for i in range(0, len(lines), 2):
        name = lines[i].replace(":", "")
        score = float(lines[i + 1])
        data[name] = score

    return data


# =========================
# Utility: check background mask (all zero)
# =========================
def is_background(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    return mask_np.max() == 0


# =========================
# Main selection function
# =========================
def select_samples(quality_path, match_path, mask_dir, save_path,
                   alpha, topk_fg, topk_bg):

    # =========================
    # Step 1: load scores
    # =========================
    quality_dict = load_score(quality_path)
    match_dict = load_score(match_path)

    # =========================
    # Step 2: normalize quality scores
    # =========================
    q_values = list(quality_dict.values())
    q_min, q_max = min(q_values), max(q_values)

    print(f"Quality score range: min={q_min}, max={q_max}")

    q_norm_dict = {}
    for k, v in quality_dict.items():
        if q_max - q_min == 0:
            q_norm = 0.0
        else:
            q_norm = (v - q_min) / (q_max - q_min)
        q_norm_dict[k] = q_norm

    # =========================
    # Step 3: split foreground / background
    # =========================
    bg_dict = {}
    fg_dict = {}

    for img_name in q_norm_dict.keys():

        base_name = img_name.replace("_0000.png", ".png")
        mask_path = os.path.join(mask_dir, base_name)

        if not os.path.exists(mask_path):
            continue

        if is_background(mask_path):
            bg_dict[img_name] = q_norm_dict[img_name]
        else:
            if img_name in match_dict:
                fg_dict[img_name] = (q_norm_dict[img_name], match_dict[img_name])

    print(f"Background samples: {len(bg_dict)}")
    print(f"Foreground samples: {len(fg_dict)}")

    # =========================
    # Step 4: select background (quality only)
    # =========================
    bg_sorted = sorted(bg_dict.items(), key=lambda x: x[1], reverse=True)
    bg_selected = bg_sorted[:topk_bg]

    # =========================
    # Step 5: select foreground (weighted score)
    # =========================
    fg_score_dict = {}
    for k, (q_norm, match) in fg_dict.items():
        S = alpha * match + (1 - alpha) * q_norm
        fg_score_dict[k] = S

    fg_sorted = sorted(fg_score_dict.items(), key=lambda x: x[1], reverse=True)
    fg_selected = fg_sorted[:topk_fg]

    # =========================
    # Step 6: save results
    # =========================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for name, score in bg_selected:
            f.write(f"{name}:{score:.6f}\n")

    with open(save_path, 'a') as f:
        for name, score in fg_selected:
            f.write(f"{name}:{score:.6f}\n")

    print(f"Background selected: {len(bg_selected)} → {save_path}")
    print(f"Foreground selected: {len(fg_selected)} → {save_path}")


# =========================
# Argument parser
# =========================
def get_args():
    parser = argparse.ArgumentParser(description="Sample selection based on quality and matching scores")

    parser.add_argument("--quality_path", type=str,
                        default="./GIQA/score_list/FeTS_control.txt")

    parser.add_argument("--match_path", type=str,
                        default="./Evaluation/FeTS_control.txt")

    parser.add_argument("--mask_dir", type=str,
                        default="./labelsTr",)

    parser.add_argument("--save_path", type=str,
                        default="./Select/FeTS/control.txt")

    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for matching score")
    parser.add_argument("--topk_fg", type=int, default=15922,help="Number of selected samples")
    parser.add_argument("--topk_bg", type=int, default=8334,help="Number of selected background")

    return parser.parse_args()


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    args = get_args()

    select_samples(
        quality_path=args.quality_path,
        match_path=args.match_path,
        mask_dir=args.mask_dir,
        save_path=args.save_path,
        alpha=args.alpha,
        topk_fg=args.topk_fg,
        topk_bg=args.topk_bg
    )