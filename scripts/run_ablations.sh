#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "Executing Full DG-ReID Component Ablation Suite"
echo "=========================================================="

OUTPUT_BASE="./outputs/ablations"
mkdir -p "$OUTPUT_BASE"

# 1. Baseline CLIP Backbone (CILP-FGDI visual baseline)
echo -e "\n[1/4] Running Baseline CLIP (No GRL, No Part Branch)..."
python train.py --config configs/default.yaml --override configs/ablations/baseline_clip.yaml --output_dir "$OUTPUT_BASE/baseline"

# 2. GRL-Only
echo -e "\n[2/4] Running GRL-Only..."
python train.py --config configs/default.yaml --override configs/ablations/grl_only.yaml --output_dir "$OUTPUT_BASE/grl_only"

# 3. Part-Only
echo -e "\n[3/4] Running Part-Only..."
python train.py --config configs/default.yaml --override configs/ablations/part_only.yaml --output_dir "$OUTPUT_BASE/part_only"

# 4. Full DG-ReID
echo -e "\n[4/4] Running Full DG-ReID..."
python train.py --config configs/default.yaml --override configs/ablations/full_dg_reid.yaml --output_dir "$OUTPUT_BASE/full_dg_reid"

echo -e "\n[+] Ablation suite training complete. Aggregating results..."
python ablate.py --mode display
