import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def main():
    # Update these paths if your data is located elsewhere
    anno_path = Path("data/trompa-coco/annotations/instances_trainval2017.json")
    summary_path = Path("experiments/026_variable_patch_size_augmentation/inference/coco_eval_summary.json")
    
    # 1. Load AP from the evaluation summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    symbol_aps = {}
    for name, stats in summary["symbols"]["per_category"].items():
        symbol_aps[name] = stats["mAP_0.5"]
        
    # 2. Load raw COCO annotations
    with open(anno_path, 'r') as f:
        coco_data = json.load(f)
        
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 3. Extract symbol scales (diagonal length) from annotations
    symbol_scales = defaultdict(list)
    for ann in coco_data['annotations']:
        cat_name = categories[ann['category_id']]
        # Only process classes that are present in our symbol evaluation summary
        if cat_name in symbol_aps:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                # Use the diagonal of the bounding box as the "scale" or "length"
                diagonal = np.sqrt(w**2 + h**2)
                symbol_scales[cat_name].append(diagonal)

    # 4. Compute variance for each class
    class_names = []
    variances = []
    aps = []
    
    for name, ap in symbol_aps.items():
        scales = symbol_scales.get(name, [])
        if len(scales) > 1:
            var = np.var(scales)
            class_names.append(name)
            variances.append(var)
            aps.append(ap)
        else:
            print(f"Warning: No valid scales found for {name}")

    # 5. Plot Variance vs AP
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(variances, aps, color='red', zorder=3)
    
    # Annotate each point with the class name
    for i, name in enumerate(class_names):
        # Only annotate some to avoid extreme clutter, or annotate all with small font
        ax.annotate(name, (variances[i], aps[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
        
    ax.set_xlabel("Variance of Symbol Diagonal Length (pixels^2)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Symbol Scale Variance vs. Detection AP")
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    plt.tight_layout()
    output_path = "symbol_variance_vs_ap.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
