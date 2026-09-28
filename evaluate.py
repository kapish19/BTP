"""
Cross-Domain and Occlusion-Robustness Evaluation Script for DG-ReID.

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/checkpoint_best.pth
    python evaluate.py --config configs/default.yaml --synthetic
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.models.dg_reid import DGReID
from src.data.dataset import ReIDImageDataset, parse_market1501_dir
from src.data.synthetic_data import generate_synthetic_reid_dataset
from src.evaluation.evaluator import Evaluator
from src.utils.config import load_config, merge_configs
from src.utils.logger import setup_logger
from src.utils.checkpoint import load_checkpoint
from src.utils.visualizer import format_occlusion_table, plot_occlusion_curves


def parse_args():
    parser = argparse.ArgumentParser(description="DG-ReID Evaluation Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--synthetic", action="store_true", help="Evaluate on synthetic target data")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")

    logger = setup_logger(log_dir=args.output_dir, log_filename="eval.log")
    logger.info(f"Evaluating DG-ReID on device: {device}")

    # Load target dataset
    if args.synthetic:
        synthetic_dir = os.path.join(args.output_dir, "synthetic_dataset")
        domain_dirs = generate_synthetic_reid_dataset(synthetic_dir, num_identities=24, images_per_identity=6)
        query_samples = parse_market1501_dir(os.path.join(domain_dirs["domain_2"], "query"), domain_id=2)
        gallery_samples = parse_market1501_dir(os.path.join(domain_dirs["domain_2"], "bounding_box_test"), domain_id=2)
        num_classes = 48
    else:
        root_dir = config["data"]["root_dir"]
        target_name = config["data"]["target_dataset"]
        query_dir = os.path.join(root_dir, target_name, "query")
        gallery_dir = os.path.join(root_dir, target_name, "bounding_box_test")
        query_samples = parse_market1501_dir(query_dir, domain_id=0)
        gallery_samples = parse_market1501_dir(gallery_dir, domain_id=0)
        num_classes = 1000

    query_dataset = ReIDImageDataset(query_samples, is_train=False)
    gallery_dataset = ReIDImageDataset(gallery_samples, is_train=False)

    batch_size = config["eval"].get("batch_size", 64)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Target Domain: Query images={len(query_dataset)} | Gallery images={len(gallery_dataset)}")

    # Build model
    model_cfg = config["model"]
    model = DGReID(
        num_classes=num_classes,
        num_domains=model_cfg.get("num_domains", 3),
        feat_dim=model_cfg.get("feat_dim", 512),
        clip_dim=model_cfg.get("clip_dim", 768),
        num_factors=model_cfg.get("num_factors", 4),
        enable_grl=model_cfg.get("enable_grl", True),
        enable_part_branch=model_cfg.get("enable_part_branch", True),
        enable_gat=model_cfg.get("enable_gat", True),
        enable_vis_gating=model_cfg.get("enable_vis_gating", True),
        enable_ortho=model_cfg.get("enable_ortho", True)
    ).to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f"Loading checkpoint weights from: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model, device=device)
    else:
        logger.warning("No checkpoint provided or found; running with initialized weights.")

    evaluator = Evaluator(
        model=model,
        device=device,
        distance_metric=config["eval"].get("distance_metric", "cosine")
    )

    severities = ["clean", "mild", "moderate", "severe"]
    results = evaluator.evaluate(query_loader, gallery_loader, occlusion_severities=severities)

    logger.info("\n" + "=" * 65)
    logger.info("EVALUATION RESULTS UNDER INCREASING OCCLUSION (TABLE 8 FORMAT)")
    logger.info("=" * 65)

    table_data = [{
        "Config": "DG-ReID (Evaluated)",
        "Clean": results["clean"]["Rank-1"],
        "Mild": results["mild"]["Rank-1"],
        "Moderate": results["moderate"]["Rank-1"],
        "Severe": results["severe"]["Rank-1"]
    }]

    print(format_occlusion_table(table_data, output_format="markdown"))
    print("\nDetailed Metrics:")
    for sev in severities:
        m = results[sev]
        print(f"[{sev.upper():<8}] Rank-1: {m['Rank-1']:5.2f}% | Rank-5: {m['Rank-5']:5.2f}% | Rank-10: {m['Rank-10']:5.2f}% | mAP: {m['mAP']:5.2f}%")

    # Plot degradation curve
    plot_path = os.path.join(args.output_dir, "occlusion_degradation.png")
    plot_occlusion_curves({"DG-ReID": results}, save_path=plot_path)


if __name__ == "__main__":
    main()
