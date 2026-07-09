import os
import argparse


# =========================
# Utility: load score file
# =========================
def load_score(file_path):
    """
    File format:
    xxx.png:
    score
    """
    data = {}

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]

    for i in range(0, len(lines), 2):
        name = lines[i].replace(":", "")
        score = float(lines[i + 1])
        data[name] = score

    return data


# =========================
# Main function
# =========================
def select_topk(quality_path, match_path, save_path, alpha, top_k):

    # =========================
    # Step 1: load data
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
    # Step 3: compute final score
    # =========================
    score_dict = {}
    common_keys = set(q_norm_dict.keys()) & set(match_dict.keys())

    for k in common_keys:
        match = match_dict[k]
        q_norm = q_norm_dict[k]
        score = alpha * match + (1 - alpha) * q_norm
        score_dict[k] = score

    # =========================
    # Step 4: top-k selection
    # =========================
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_items[:top_k]

    # =========================
    # Step 5: save results
    # =========================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for name, score in selected:
            f.write(f"{name}:{score:.6f}\n")

    print(f"Selected {len(selected)} images.")
    print(f"Results saved to: {save_path}")


# =========================
# Argument parser
# =========================
def get_args():
    parser = argparse.ArgumentParser(description="Top-K sample selection using quality and matching scores")

    parser.add_argument("--quality_path", type=str,
                        default="./GIQA/score_list/FeTS_control.txt")

    parser.add_argument("--match_path", type=str,
                        default="./Evaluation/FeTS_control.txt")

    parser.add_argument("--save_path", type=str,
                        default="./Select/Nuclei/control.txt")

    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for matching score")

    parser.add_argument("--top_k", type=int, default=713,
                        help="Number of selected samples")

    return parser.parse_args()


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    args = get_args()

    select_topk(
        quality_path=args.quality_path,
        match_path=args.match_path,
        save_path=args.save_path,
        alpha=args.alpha,
        top_k=args.top_k
    )