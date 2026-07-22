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
    
    line_aps = {}
    for name, stats in summary["lines"]["per_category"].items():
        line_aps[name] = stats["mAP_0.5"]
        
    # 2. Load raw COCO annotations
    with open(anno_path, 'r') as f:
        coco_data = json.load(f)
        
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 3. Extract line lengths from annotations
    line_lengths = defaultdict(list)
    for ann in coco_data['annotations']:
        cat_name = categories[ann['category_id']]
        # Only process classes that are present in our line evaluation summary
        if cat_name in line_aps:
            # Trompa-COCO represents lines as keypoints: [x1, y1, v1, x2, y2, v2]
            if 'keypoints' in ann and len(ann['keypoints']) >= 4:
                kp = ann['keypoints']
                x1, y1, v1, x2, y2, v2 = kp
                # Only calculate length if both keypoints are visible/valid
                if v1 > 0 and v2 > 0:
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    line_lengths[cat_name].append(length)
            elif 'bbox' in ann:
                # Fallback to bbox diagonal if keypoints are missing
                x, y, w, h = ann['bbox']
                length = np.sqrt(w**2 + h**2)
                line_lengths[cat_name].append(length)

    # 4. Compute variance for each class
    class_names = []
    variances = []
    aps = []
    
    for name, ap in line_aps.items():
        lengths = line_lengths.get(name, [])
        if len(lengths) > 1:
            var = np.var(lengths)
            class_names.append(name)
            variances.append(var)
            aps.append(ap)
        else:
            print(f"Warning: No valid lengths found for {name}")

    # 5. Plot Variance vs AP
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(variances, aps, color='blue', zorder=3)
    
    # Annotate each point with the class name
    for i, name in enumerate(class_names):
        ax.annotate(name, (variances[i], aps[i]), textcoords="offset points", xytext=(5,5), ha='center')
        
    ax.set_xlabel("Variance of Line Length (pixels^2)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Line Length Variance vs. Detection AP")
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    plt.tight_layout()
    output_path = "line_variance_vs_ap.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
