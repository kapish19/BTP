"""
Automated Ablation Study Runner for DG-ReID.
Executes the controlled component-wise ablations reported in Tables 7 & 8 of the BTP report:

Table 7 (Component Ablation):
  1. CILP-FGDI Baseline:  disable_grl=True,  disable_part_branch=True,  eval_feat='global'
  2. GRL-Only:           disable_grl=False, disable_part_branch=True,  eval_feat='global'
  3. Part-Only:          disable_grl=True,  disable_part_branch=False, eval_feat='combined'
  4. Full DG-ReID:       disable_grl=False, disable_part_branch=False, eval_feat='combined'

Table 8 (Target-Domain Occlusion Robustness):
  - Global Baseline vs Part Branch (No Vis) vs Full DG-ReID
  - Across: Clean (0%), Mild (15%), Moderate (30%), Severe (50%) occlusion
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


ABLATION_CONFIGS = [
    {
        "name": "CILP-FGDI Baseline",
        "flags": ["--disable_grl", "--disable_part_branch", "--eval_feat", "global"],
        "desc": "Baseline CLIP backbone without GRL and without Part branch"
    },
    {
        "name": "GRL-Only",
        "flags": ["--disable_part_branch", "--eval_feat", "global"],
        "desc": "Visual-side GRL domain erasure without Part branch"
    },
    {
        "name": "Part-Only",
        "flags": ["--disable_grl", "--eval_feat", "combined"],
        "desc": "PartVisibilityGAT branch without visual GRL"
    },
    {
        "name": "Full DG-ReID",
        "flags": ["--eval_feat", "combined"],
        "desc": "Full proposed DG-ReID architecture combining GRL and PartVisibilityGAT"
    }
]

OCCLUSION_BENCHMARKS = [
    {
        "model_name": "Global Baseline",
        "flags": ["--disable_grl", "--disable_part_branch", "--eval_feat", "global"]
    },
    {
        "model_name": "+ Part Branch (No Visibility)",
        "flags": ["--disable_grl", "--disable_vis_gating", "--eval_feat", "combined"]
    },
    {
        "model_name": "Full DG-ReID",
        "flags": ["--eval_feat", "combined"]
    }
]

SEVERITIES = ["clean", "mild", "moderate", "severe"]


def run_experiment(args, extra_flags, exp_tag):
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "run.py"),
        "--benchmark", args.benchmark,
        "--held_out_domain", args.held_out_domain,
        "--data_path", args.data_path,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--log_path", os.path.join(args.output_dir, exp_tag)
    ] + extra_flags

    if args.debug_subset_ids > 0:
        cmd += ["--debug_subset_ids", str(args.debug_subset_ids)]

    print(f"\n=======================================================")
    print(f"🚀 Running: {exp_tag}")
    print(f"Command: {' '.join(cmd)}")
    print(f"=======================================================\n")

    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run DG-ReID Ablation Studies")
    parser.add_argument('--benchmark', type=str, default='protocol2')
    parser.add_argument('--held_out_domain', type=str, default='Market')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='ablation_results')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--debug_subset_ids', type=int, default=0,
                        help="Subsample PIDs for rapid testing")
    parser.add_argument('--run_occlusion', action='store_true',
                        help="Also run target occlusion robustness benchmarks (Table 8)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_file = os.path.join(args.output_dir, "ablation_summary.json")

    print(f"Starting DG-ReID Ablations on {args.held_out_domain}...")

    # 1. Run Table 7 Component Ablations
    for cfg in ABLATION_CONFIGS:
        tag = f"ablation_{cfg['name'].lower().replace(' ', '_').replace('-', '_')}"
        run_experiment(args, cfg['flags'], tag)

    # 2. Optionally Run Table 8 Occlusion Benchmarks
    if args.run_occlusion:
        print("\n--- Running Target Occlusion Robustness Experiments (Table 8) ---")
        for model_cfg in OCCLUSION_BENCHMARKS:
            for sev in SEVERITIES:
                tag = f"occ_{model_cfg['model_name'].lower().replace(' ', '_')}_{sev}"
                extra = model_cfg['flags'] + ["--occlusion_severity", sev]
                run_experiment(args, extra, tag)

    print("\n✅ All requested ablation runs finished successfully!")


if __name__ == "__main__":
    main()
