"""
DG-ReID: Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification.
Main Execution Engine supporting:
  - 3-Stage Training (Table 3):
      Stage 1 (Epochs 1-10):  Freeze backbone, train heads, GAT, visibility, domain factors.
      Stage 2 (Epochs 11-30): Unfreeze final 2 ViT transformer blocks.
      Stage 3 (Epochs 31-60): Domain classifier with GRL warm-up alpha(e) = min(1, (e-30)/10).
  - 7-Component Joint Optimization Objective (Eq. 1):
      L_total = lambda_id*L_id^g + lambda_tri*L_tri^g + lambda_id_l*L_id^l + lambda_tri_l*L_tri^l
              + lambda_occ*L_occ + lambda_adv*L_dom + lambda_ortho*L_ortho.
  - Zero-shot Cross-domain Evaluation with Market-1501 protocol (mAP, Rank-1, Rank-5, Rank-10).
  - Target-query Occlusion Robustness Evaluation (Clean, Mild, Moderate, Severe).
  - Full Ablation Suite (Baseline CLIP, GRL-Only, Part-Only, Full DG-ReID).
  - Checkpoint saving, best model selection, and automatic resumption.
"""

import os
import sys
import time
import argparse
import datetime
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cfgs import protocol_1, protocol_2, protocol_occluded_duke
from models import DGReID, BaselineCLIP, compute_grl_alpha
from loss import make_loss
from solver import build_optimizer, WarmupMultiStepLR, StageManager
from datasets.build import build_data_loader
from datasets.occlusion import apply_target_occlusion
from datasets import DATASET_REGISTRY
from reidutils.meter import AverageMeter
from reidutils.metrics import R1_mAP_eval
from reidutils.logger import setup_logger
from reidutils.checkpoint import CheckpointManager


def get_device(args):
    req_device = getattr(args, 'device', 'auto')
    if req_device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(req_device)


def build_model(args, num_classes, num_domains):
    device = get_device(args)
    model = DGReID(
        num_classes=num_classes,
        num_domains=num_domains,
        backbone_name=args.backbone,
        img_size=args.size_train,
        feat_dim=512,
        num_factors=4,
        pretrained=True
    ).to(device)
    return model


def train_one_epoch(
    epoch,
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device,
    stage,
    grl_alpha,
    args,
    logger,
    meters
):
    model.train()
    start_time = time.time()
    loss_meter = meters['total_loss']
    loss_meter.reset()

    use_amp = (device.type == 'cuda')

    for n_iter, (img, pids, camids, viewids, domains, cids, occ_masks) in enumerate(train_loader):
        optimizer.zero_grad()

        img = img.to(device)
        targets = pids.to(device)
        domain_labels = domains.to(device)
        occ_targets = occ_masks.to(device)

        if use_amp:
            with amp.autocast(enabled=True):
                outputs = model(
                    x=img,
                    label=targets,
                    grl_alpha=grl_alpha,
                    disable_grl=args.disable_grl,
                    disable_part_branch=args.disable_part_branch,
                    disable_vis_gating=args.disable_vis_gating
                )
                loss, loss_dict = criterion(
                    outputs,
                    targets,
                    domain_labels=domain_labels,
                    occ_targets=occ_targets,
                    stage=stage,
                    grl_alpha=grl_alpha,
                    disable_grl=args.disable_grl,
                    disable_part_branch=args.disable_part_branch
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                x=img,
                label=targets,
                grl_alpha=grl_alpha,
                disable_grl=args.disable_grl,
                disable_part_branch=args.disable_part_branch,
                disable_vis_gating=args.disable_vis_gating
            )
            loss, loss_dict = criterion(
                outputs,
                targets,
                domain_labels=domain_labels,
                occ_targets=occ_targets,
                stage=stage,
                grl_alpha=grl_alpha,
                disable_grl=args.disable_grl,
                disable_part_branch=args.disable_part_branch
            )
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), img.shape[0])

        if (n_iter + 1) % args.log_period == 0 or (n_iter + 1) == len(train_loader):
            lrs = [group['lr'] for group in optimizer.param_groups]
            base_lr = min(lrs)
            head_lr = max(lrs)
            logger.info(
                f"[Stage {stage}] Epoch[{epoch:02d}] Iter[{n_iter+1:03d}/{len(train_loader):03d}] "
                f"Loss: {loss_meter.avg:.4f} "
                f"(id_g: {loss_dict['loss_id_global']:.3f}, tri_g: {loss_dict['loss_tri_global']:.3f}, "
                f"id_l: {loss_dict['loss_id_local']:.3f}, tri_l: {loss_dict['loss_tri_local']:.3f}, "
                f"occ: {loss_dict['loss_occ']:.3f}, dom: {loss_dict['loss_dom']:.3f}, ortho: {loss_dict['loss_ortho']:.4f}) | "
                f"GRL α: {grl_alpha:.2f} | lr_base: {base_lr:.2e}, lr_head: {head_lr:.2e}"
            )

    elapsed = time.time() - start_time
    logger.info(f"Epoch {epoch} finished in {elapsed:.1f}s. Average Loss: {loss_meter.avg:.4f}")
    return loss_meter.avg


def evaluate(
    val_loaders,
    model,
    device,
    logger,
    eval_feat="combined",
    occlusion_severity="clean",
    disable_part_branch=False
):
    model.eval()
    results = {}

    for name, val_loader_data in val_loaders.items():
        loader, num_query = val_loader_data
        evaluator = R1_mAP_eval(num_query, max_rank=10, feat_norm=True, reranking=False)
        evaluator.reset()

        logger.info(f"Evaluating target: {name} (Feature: {eval_feat}, Occlusion: {occlusion_severity})")

        for n_iter, (img, pids, camids, viewids, domains, cids, _) in enumerate(loader):
            with torch.no_grad():
                # Apply deterministic target occlusion if specified (Table 8)
                if occlusion_severity != "clean":
                    img_occ = torch.stack([
                        apply_target_occlusion(img[i], severity=occlusion_severity, seed=i)
                        for i in range(img.shape[0])
                    ], dim=0)
                    img = img_occ

                img = img.to(device)
                feat = model(
                    img,
                    disable_part_branch=disable_part_branch,
                    eval_feat=eval_feat
                )
                evaluator.update((feat, pids, camids))

        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info(f"[{name}] Results: mAP: {mAP:.1%}, Rank-1: {cmc[0]:.1%}, Rank-5: {cmc[4]:.1%}, Rank-10: {cmc[9]:.1%}")

        results[name] = {
            'mAP': float(mAP),
            'Rank-1': float(cmc[0]),
            'Rank-5': float(cmc[4]),
            'Rank-10': float(cmc[9]),
            'CMC': cmc[:10].tolist()
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="DG-ReID Training and Evaluation Engine")
    parser_test = argparse.ArgumentParser(description="Test Parser")

    temp_args, _ = parser.parse_known_args()
    benchmark = getattr(temp_args, 'benchmark', 'protocol2')
    held_out_domain = getattr(temp_args, 'held_out_domain', 'Market')
    no_sysu = getattr(temp_args, 'no_sysu', False)

    if benchmark == 'protocol2':
        parsertrain, parsertest, logname = protocol_2(parser, parser_test, held_out_domain, no_sysu=no_sysu)
    else:
        parsertrain, parsertest, logname = protocol_occluded_duke(parser, parser_test)

    args_train = parsertrain.parse_known_args()[0]
    args_test = parsertest.parse_known_args()[0]

    device = get_device(args_train)

    # Filter out empty datasets if any
    valid_train = []
    valid_classes = []
    for d, c in zip(args_train.train_datasets, args_train.classes):
        ds = DATASET_REGISTRY.get(d)(root=args_train.data_path, verbose=False)
        if len(ds.train) > 0:
            valid_train.append(d)
            valid_classes.append(c)
        else:
            print(f"⚠️ Dataset '{d}' returned 0 training images on disk. Excluding from source domains.")

    args_train.train_datasets = valid_train
    args_train.classes = valid_classes

    # Setup directories and logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_suffix = f"{logname}_{args_train.backbone}_{timestamp}"
    log_dir = os.path.join(args_train.log_path, exp_suffix)
    os.makedirs(log_dir, exist_ok=True)

    logger_train = setup_logger("DG-ReID_train", log_dir, if_train=True)
    logger_test = setup_logger("DG-ReID_test", log_dir, if_train=False)

    logger_train.info(f"Target Device: {device}")
    logger_train.info(f"Training Sources: {args_train.train_datasets} (Classes: {args_train.classes})")
    logger_train.info(f"Unseen Target: {args_test.test_datasets}")
    logger_train.info(f"Model: DG-ReID ({args_train.backbone}), Resolution: {args_train.size_train}")

    # Build Dataloaders
    train_loader_stage1, train_loader_stage2, val_loaders = build_data_loader(args_train, args_test)

    # Build Model
    total_classes = sum(args_train.classes) if len(args_train.classes) > 0 else 100
    num_domains = max(1, len(args_train.train_datasets))
    model = build_model(args_train, num_classes=total_classes, num_domains=num_domains)

    # Loss criterion
    criterion = make_loss(total_classes, args_train)

    # Optimizer and Scheduler
    optimizer = build_optimizer(model, args_train)
    scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=[30, 50],
        gamma=0.1,
        warmup_factor=0.1,
        warmup_iters=10,
        warmup_method='linear'
    )

    # Checkpoint and Stage Managers
    ckpt_mgr = CheckpointManager(log_dir, logger=logger_train)
    stage_mgr = StageManager(
        model,
        args_train,
        stage1_epochs=args_train.stage1_epochs,
        stage2_epochs=args_train.stage2_epochs,
        total_epochs=args_train.epochs
    )

    # Resume from checkpoint if requested
    start_epoch = 1
    resume_path = getattr(args_train, 'resume', '')
    if resume_path and os.path.isfile(resume_path):
        start_epoch = ckpt_mgr.resume(resume_path, model, optimizer, scheduler)

    # If eval_only, run evaluation and exit
    if getattr(args_train, 'eval_only', False):
        logger_train.info("⚡ Running evaluation only (--eval_only)...")
        results = evaluate(
            val_loaders,
            model,
            device,
            logger_test,
            eval_feat=args_train.eval_feat,
            occlusion_severity=args_train.occlusion_severity,
            disable_part_branch=args_train.disable_part_branch
        )
        return results

    # Training Loop
    logger_train.info(f"🚀 Starting 3-Stage Training Pipeline (Total Epochs: {args_train.epochs})")
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    meters = {'total_loss': AverageMeter()}

    for epoch in range(start_epoch, args_train.epochs + 1):
        # Update stage & parameter freezing
        stage, grl_alpha = stage_mgr.update_stage(epoch, optimizer=optimizer, logger=logger_train)
        scheduler.step()

        # Choose appropriate loader (Stage 1 can use normal loader or PK sampler)
        loader = train_loader_stage2

        loss_avg = train_one_epoch(
            epoch,
            model,
            loader,
            criterion,
            optimizer,
            scaler,
            device,
            stage,
            grl_alpha,
            args_train,
            logger_train,
            meters
        )

        # Periodic Evaluation
        if epoch % args_train.eval_period == 0 or epoch == args_train.epochs:
            val_res = evaluate(
                val_loaders,
                model,
                device,
                logger_test,
                eval_feat=args_train.eval_feat,
                occlusion_severity=args_train.occlusion_severity,
                disable_part_branch=args_train.disable_part_branch
            )
            # Track best model based on first test domain mAP
            first_target = list(val_res.keys())[0]
            target_map = val_res[first_target]['mAP']
            target_r1 = val_res[first_target]['Rank-1']
            is_best = ckpt_mgr.update_best(target_map, target_r1, epoch)
            ckpt_mgr.save(epoch, model, optimizer, scheduler, metrics=val_res, is_best=is_best)

        elif epoch % args_train.checkpoint_period == 0:
            ckpt_mgr.save(epoch, model, optimizer, scheduler)

    logger_train.info("🎉 Full 3-Stage Training Pipeline Completed Successfully!")

    # Final evaluation
    final_res = evaluate(
        val_loaders,
        model,
        device,
        logger_test,
        eval_feat=args_train.eval_feat,
        occlusion_severity=args_train.occlusion_severity,
        disable_part_branch=args_train.disable_part_branch
    )
    return final_res


if __name__ == "__main__":
    main()
