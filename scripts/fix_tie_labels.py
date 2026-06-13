import json
from pathlib import Path

def main() -> None:
    anno_path = Path("data/trompa-coco/annotations/instances_trainval2017.json")
    
    print(f"Loading JSON from {anno_path}...")
    with open(anno_path, "r") as f:
        data = json.load(f)

    # 1. Find the correct 'tie' category ID
    tie_cat_id: int | None = None
    for cat in data["categories"]:
        if cat["name"] == "tie":
            tie_cat_id = cat["id"]
            break
            
    if tie_cat_id is None:
        print("Error: Base 'tie' category not found in JSON!")
        return

    # 2. Identify buggy tie categories
    buggy_tie_ids: set[int] = set()
    valid_categories: list[dict] = []
    
    for cat in data["categories"]:
        if cat["name"].startswith("tie ") and cat["name"] != "tie":
            buggy_tie_ids.add(cat["id"])
        else:
            valid_categories.append(cat)

    print(f"Found {len(buggy_tie_ids)} buggy tie categories to merge.")

    # 3. Update annotations
    updated_count = 0
    for ann in data["annotations"]:
        if ann["category_id"] in buggy_tie_ids:
            ann["category_id"] = tie_cat_id
            updated_count += 1

    print(f"Reassigned {updated_count} annotations to the base 'tie' category.")

    # 4. Update the categories list
    data["categories"] = valid_categories

    # 5. Save the fixed JSON
    print("Saving fixed JSON...")
    with open(anno_path, "w") as f:
        json.dump(data, f)
        
    print("Done! Remember to manually delete the .pkl cache file so the dataset parser picks up the changes.")

if __name__ == "__main__":
    main()
