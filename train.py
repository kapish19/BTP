"""
Unified Training Script for DG-ReID:
Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/market1501_dg.yaml --stage 1
    python train.py --config configs/default.yaml --synthetic
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.models.dg_reid import DGReID
from src.data.dataset import ReIDImageDataset, MultiDomainReIDDataset, parse_market1501_dir, load_reid_benchmark
from src.data.sampler import MultiDomainPKSampler
from src.data.synthetic_data import generate_synthetic_reid_dataset
from src.data.occlusion import SyntheticOcclusionGenerator
from src.training.trainer import Trainer
from src.utils.config import load_config, merge_configs
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed, get_environment_info
from src.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="DG-ReID Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--override", type=str, default=None, help="Optional ablation config to merge")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2, 3], help="Train specific stage only")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for verification")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory for logs and weights")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--workers", type=int, default=None, help="Number of DataLoader workers (default: from config)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.override:
        override_cfg = load_config(args.override)
        config = merge_configs(config, override_cfg)

    # Seed and Environment
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed, deterministic=config.get("deterministic", True))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")

    logger = setup_logger(log_dir=os.path.join(args.output_dir, "logs"))
    logger.info(f"Using compute device: {device}")

    # Dataset loading
    occ_cfg = config.get("data", {}).get("occlusion", {})
    occ_gen = SyntheticOcclusionGenerator(
        img_size=tuple(config["model"]["img_size"]),
        patch_size=config["model"]["patch_size"],
        prob=occ_cfg.get("prob", 0.5),
        area_range=tuple(occ_cfg.get("area_range", [0.1, 0.5])),
        aspect_range=tuple(occ_cfg.get("aspect_range", [0.3, 3.3]))
    )

    if args.synthetic:
        logger.info("[*] Generating synthetic multi-domain dataset for pipeline verification...")
        synthetic_dir = os.path.join(args.output_dir, "synthetic_dataset")
        domain_dirs = generate_synthetic_reid_dataset(synthetic_dir, num_identities=24, images_per_identity=6)

        # Source domains: domain_0, domain_1
        source_data = [
            ("domain_0", parse_market1501_dir(os.path.join(domain_dirs["domain_0"], "bounding_box_train"), domain_id=0)),
            ("domain_1", parse_market1501_dir(os.path.join(domain_dirs["domain_1"], "bounding_box_train"), domain_id=1))
        ]
        train_dataset = MultiDomainReIDDataset(source_data, occlusion_generator=occ_gen)

        # Target domain: domain_2
        target_query = parse_market1501_dir(os.path.join(domain_dirs["domain_2"], "query"), domain_id=2)
        target_gallery = parse_market1501_dir(os.path.join(domain_dirs["domain_2"], "bounding_box_test"), domain_id=2)

        val_q_dataset = ReIDImageDataset(target_query, is_train=False)
        val_g_dataset = ReIDImageDataset(target_gallery, is_train=False)
        num_classes = train_dataset.num_classes
        num_domains = 2

    else:
        root_dir = config["data"]["root_dir"]
        source_names = config["data"]["source_datasets"]
        target_name = config["data"]["target_dataset"]

        logger.info(f"Sources: {source_names} -> Target: {target_name}")

        source_data = []
        for idx, s_name in enumerate(source_names):
            s_train, _, _ = load_reid_benchmark(root_dir, s_name, domain_id=idx)
            logger.info(f"Loaded {s_name}: {len(s_train)} training images")
            source_data.append((s_name, s_train))

        train_dataset = MultiDomainReIDDataset(source_data, occlusion_generator=occ_gen)

        _, target_query, target_gallery = load_reid_benchmark(root_dir, target_name, domain_id=len(source_names))
        logger.info(f"Loaded target {target_name}: {len(target_query)} query images, {len(target_gallery)} gallery images")
        val_q_dataset = ReIDImageDataset(target_query, is_train=False)
        val_g_dataset = ReIDImageDataset(target_gallery, is_train=False)
        num_classes = train_dataset.num_classes
        num_domains = len(source_names)

    # Sampler
    p = config["data"].get("p", 16)
    k = config["data"].get("k", 4)
    # Adjust P if synthetic identities are fewer
    if train_dataset.num_classes < p:
        p = max(2, train_dataset.num_classes // 2)

    num_workers = args.workers if args.workers is not None else config["data"].get("num_workers", 0)
    sampler = MultiDomainPKSampler(train_dataset, p=p, k=k, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=num_workers)

    val_q_loader = DataLoader(val_q_dataset, batch_size=config["eval"].get("batch_size", 64), shuffle=False) if len(val_q_dataset) > 0 else None
    val_g_loader = DataLoader(val_g_dataset, batch_size=config["eval"].get("batch_size", 64), shuffle=False) if len(val_g_dataset) > 0 else None

    # Model
    model_cfg = config["model"]
    model = DGReID(
        num_classes=num_classes,
        num_domains=num_domains,
        feat_dim=model_cfg.get("feat_dim", 512),
        clip_dim=model_cfg.get("clip_dim", 768),
        num_factors=model_cfg.get("num_factors", 4),
        unfreeze_blocks=model_cfg.get("unfreeze_blocks", 2),
        enable_grl=model_cfg.get("enable_grl", True),
        enable_part_branch=model_cfg.get("enable_part_branch", True),
        enable_gat=model_cfg.get("enable_gat", True),
        enable_vis_gating=model_cfg.get("enable_vis_gating", True),
        enable_ortho=model_cfg.get("enable_ortho", True),
        enable_reconstruction_gate=model_cfg.get("enable_reconstruction_gate", True)
    )

    if args.resume:
        logger.info(f"Resuming checkpoint from: {args.resume}")
        load_checkpoint(args.resume, model, device=device)

    # Override training epochs if requested
    if args.epochs is not None:
        config["train"]["total_epochs"] = args.epochs
    if args.stage is not None:
        if args.stage == 1:
            config["train"]["total_epochs"] = config["train"]["stage1_epochs"]
        elif args.stage == 2:
            config["train"]["total_epochs"] = config["train"]["stage2_epochs"]

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_query_loader=val_q_loader,
        val_gallery_loader=val_g_loader,
        config=config,
        device=device,
        output_dir=args.output_dir,
        logger=logger
    )

    trainer.run()


if __name__ == "__main__":
    main()
