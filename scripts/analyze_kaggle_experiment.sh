#!/bin/bash
set -e

# Hardcoded parameters
STATE_FILE=".current_experiment"
KAGGLE_SLUG="kwonyoungchoi/music-deep"
ANNO_PATH="data/trompa-coco/annotations/instances_trainval2017.json"
STAGE_NAME="train_detection" # matches the default --stage_name in train_detection.py

# Check if the state file exists
if [ ! -f "$STATE_FILE" ]; then
    echo "Error: State file '$STATE_FILE' not found."
    echo "Please write the target experiment name to '$STATE_FILE' first."
    echo "Example: echo '018_affine_augmentation' > $STATE_FILE"
    exit 1
fi

# Read the current experiment name from the file
EXP_NAME=$(cat "$STATE_FILE")
EXP_DIR="experiments/$EXP_NAME"

echo "=== Current Experiment: $EXP_NAME ==="
mkdir -p "$EXP_DIR"

echo "=== Step 4: Downloading Kaggle Outputs to $EXP_DIR ==="
mamba run -n pytorch kaggle kernels output "$KAGGLE_SLUG" -p "$EXP_DIR"

echo "=== Step 5: Running COCO Evaluation ==="
PRED_DIR="$EXP_DIR/inference"
if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Prediction directory $PRED_DIR does not exist."
    exit 1
fi

mamba run -n pytorch python src/evaluate_coco.py \
    --anno_path "$ANNO_PATH" \
    --pred_dir "$PRED_DIR" \
    --out_dir "$EXP_DIR"

EVAL_METRICS="$EXP_DIR/coco_eval_summary.json"
TRAIN_METRICS="$EXP_DIR/$STAGE_NAME/metrics.jsonl"

echo "=== Step 6: Summarizing Results with Aider ==="
echo "Found training metrics: $TRAIN_METRICS"
echo "Found inference metrics: $EVAL_METRICS"

AIDER_READ_ARGS=""
if [ -f "$TRAIN_METRICS" ]; then
    AIDER_READ_ARGS="$AIDER_READ_ARGS --read $TRAIN_METRICS"
else
    echo "Warning: Training metrics not found at $TRAIN_METRICS"
fi

if [ -f "$EVAL_METRICS" ]; then
    AIDER_READ_ARGS="$AIDER_READ_ARGS --read $EVAL_METRICS"
else
    echo "Warning: Inference metrics not found at $EVAL_METRICS"
fi

MESSAGE="I have downloaded the latest Kaggle outputs for the experiment located at \`$EXP_DIR\`. 
I have attached the training metrics and the official COCO evaluation metrics as read-only context files.
Please find the corresponding experiment section in EXPERIMENTS.md (it should match the folder name \`$EXP_NAME\` or the directory \`$EXP_DIR\`).
Analyze the provided metrics files to summarize the training dynamics (losses, in-training mAP) and the official COCO evaluation results (global_stats, per_category mAP_0.5).
Update the 'Results' subsection with a concise summary of these metrics.
Update the 'Conclusion' subsection with a brief, pedagogical insight into what these results mean for the model's learning and what the next steps should be.
Do not modify any other experiments or parts of the file."

aider --yes --no-clipboard --message "$MESSAGE" $AIDER_READ_ARGS EXPERIMENTS.md

echo "=== Automation Complete ==="
