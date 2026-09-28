#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "Executing Full 3-Stage Training Pipeline for DG-ReID"
echo "=========================================================="

CONFIG=${1:-"configs/default.yaml"}
OUTPUT_DIR=${2:-"./outputs/full_run"}

echo "Using Configuration: $CONFIG"
echo "Output Directory:    $OUTPUT_DIR"

# Stage 1: Epochs 1-10 (Frozen Backbone)
echo -e "\n[+] Launching Stage 1..."
python train.py --config "$CONFIG" --stage 1 --output_dir "$OUTPUT_DIR"

# Stage 2: Epochs 11-30 (Final 2 ViT blocks unfrozen)
echo -e "\n[+] Launching Stage 2..."
python train.py --config "$CONFIG" --stage 2 --resume "$OUTPUT_DIR/checkpoints/stage_1_final.pth" --output_dir "$OUTPUT_DIR"

# Stage 3: Epochs 31-60 (Adversarial GRL with warmup)
echo -e "\n[+] Launching Stage 3..."
python train.py --config "$CONFIG" --stage 3 --resume "$OUTPUT_DIR/checkpoints/stage_2_final.pth" --output_dir "$OUTPUT_DIR"

echo -e "\n[+] All 3 stages completed successfully!"
echo "[+] Final checkpoint: $OUTPUT_DIR/checkpoints/checkpoint_best.pth"
