"""
Full research-grade training engine supporting:
- 3-stage optimization with automatic backbone freezing/unfreezing
- Mixed precision (AMP)
- Differential learning rates for backbone vs heads
- Checkpoint management and validation monitoring
- CSV and optional TensorBoard loss logging
"""

import csv
import os
import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .stage_manager import StageManager
from ..losses.total_loss import TotalLoss
from ..evaluation.evaluator import Evaluator
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.logger import setup_logger
from ..utils.reproducibility import get_environment_info


class Trainer:
    """
    Unified Trainer for DG-ReID across Stages 1, 2, and 3.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_query_loader: Optional[DataLoader],
        val_gallery_loader: Optional[DataLoader],
        config: Dict[str, Any],
        device: torch.device,
        output_dir: str = "./outputs",
        logger: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_query_loader = val_query_loader
        self.val_gallery_loader = val_gallery_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logger if logger is not None else setup_logger(log_dir=self.log_dir)

        # Loss function
        loss_cfg = config.get("loss", {})
        self.criterion = TotalLoss(
            lambda_id=loss_cfg.get("lambda_id", 1.0),
            lambda_tri=loss_cfg.get("lambda_tri", 1.0),
            lambda_id_loc=loss_cfg.get("lambda_id_loc", 0.5),
            lambda_tri_loc=loss_cfg.get("lambda_tri_loc", 0.5),
            lambda_occ=loss_cfg.get("lambda_occ", 0.3),
            lambda_adv=loss_cfg.get("lambda_adv", 0.1),
            lambda_ortho=loss_cfg.get("lambda_ortho", 0.05),
            triplet_margin=loss_cfg.get("triplet_margin", 0.3),
            label_smooth_eps=loss_cfg.get("label_smooth_eps", 0.1)
        ).to(device)

        # Stage manager
        train_cfg = config.get("train", {})
        self.stage_manager = StageManager(
            stage1_epochs=train_cfg.get("stage1_epochs", 10),
            stage2_epochs=train_cfg.get("stage2_epochs", 30),
            stage3_epochs=train_cfg.get("stage3_epochs", 60),
            grl_ramp_epochs=train_cfg.get("grl_ramp_epochs", 10)
        )

        # Optimizer and Scheduler
        self.lr_head = float(train_cfg.get("lr_head", 3.5e-4))
        self.lr_backbone = float(train_cfg.get("lr_backbone", 3.5e-5))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.total_epochs = train_cfg.get("total_epochs", 60)
        self.eval_period = train_cfg.get("eval_period", 5)
        self.clip_grad_norm = train_cfg.get("clip_grad_norm", 1.0)
        self.use_amp = train_cfg.get("use_amp", True) and torch.cuda.is_available()

        # start_epoch must be set before _build_optimizer_and_scheduler uses it
        self.start_epoch = 1

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()

        # Best metrics tracking
        self.best_rank1 = 0.0
        self.best_map = 0.0

        # CSV Logging
        self.csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        self._init_csv()

    def _build_optimizer_and_scheduler(self):
        """
        Creates parameter groups with differential learning rates for backbone vs heads.
        """
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if head_params:
            param_groups.append({'params': head_params, 'lr': self.lr_head, 'weight_decay': self.weight_decay})
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': self.lr_backbone, 'weight_decay': self.weight_decay})

        # Fallback if no params have requires_grad yet (Stage 1 initialization)
        if not param_groups:
            param_groups = [{'params': self.model.parameters(), 'lr': self.lr_head, 'weight_decay': self.weight_decay}]

        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.total_epochs, eta_min=1e-6,
            last_epoch=self.start_epoch - 2  # avoids "step before optimizer.step()" warning
        )
        return optimizer, scheduler

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "stage", "lr_head", "loss_total",
                    "loss_id_g", "loss_tri_g", "loss_id_l", "loss_tri_l",
                    "loss_occ", "loss_dom", "loss_ortho", "rank1", "map"
                ])

    def train_epoch(self, epoch: int, stage: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {}
        batch_count = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            imgs = batch[0].to(self.device)
            pids = batch[1].to(self.device)
            camids = batch[2].to(self.device)
            domain_ids = batch[3].to(self.device)
            vis_targets = batch[4].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(imgs, stage=stage)
                    loss, loss_dict = self.criterion(
                        outputs=outputs,
                        pids=pids,
                        domain_labels=domain_ids,
                        vis_targets=vis_targets,
                        stage=stage
                    )
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(imgs, stage=stage)
                loss, loss_dict = self.criterion(
                    outputs=outputs,
                    pids=pids,
                    domain_labels=domain_ids,
                    vis_targets=vis_targets,
                    stage=stage
                )
                loss.backward()
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

            # Accumulate loss metrics
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"  [!] NaN/Inf loss at batch {batch_idx}, skipping.")
                self.optimizer.zero_grad()
                continue

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            batch_count += 1

            # Log live progress every 100 batches
            total_batches = len(self.train_loader)
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                step_elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / max(1e-3, step_elapsed)
                self.logger.info(
                    f"Epoch [{epoch}/{self.total_epochs}] Iter [{batch_idx + 1:4d}/{total_batches:4d}] "
                    f"({(batch_idx + 1) / total_batches * 100:5.1f}%) | "
                    f"Loss: {loss.item():.4f} | "
                    f"ID: {loss_dict.get('loss_id_g', 0.0):.3f} | "
                    f"Tri: {loss_dict.get('loss_tri_g', 0.0):.3f} | "
                    f"Occ: {loss_dict.get('loss_occ', 0.0):.3f} | "
                    f"Speed: {batches_per_sec:.1f} b/s"
                )

        self.scheduler.step()
        elapsed = time.time() - start_time

        # Average losses
        avg_losses = {k: v / max(1, batch_count) for k, v in epoch_losses.items()}
        current_lr = self.optimizer.param_groups[0]['lr']

        self.logger.info(
            f"Epoch [{epoch}/{self.total_epochs}] Stage {stage} | "
            f"Loss: {avg_losses.get('loss_total', 0.0):.4f} | "
            f"ID_g: {avg_losses.get('loss_id_g', 0.0):.3f} | "
            f"Tri_g: {avg_losses.get('loss_tri_g', 0.0):.3f} | "
            f"Occ: {avg_losses.get('loss_occ', 0.0):.3f} | "
            f"Dom: {avg_losses.get('loss_dom', 0.0):.3f} | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )
        return avg_losses

    def evaluate_model(self) -> Dict[str, float]:
        """
        Runs clean target evaluation on held-out query and gallery.
        """
        if self.val_query_loader is None or self.val_gallery_loader is None:
            return {"Rank-1": 0.0, "mAP": 0.0}

        evaluator = Evaluator(model=self.model, device=self.device)
        res = evaluator.evaluate(
            self.val_query_loader,
            self.val_gallery_loader,
            occlusion_severities=["clean"]
        )
        clean_res = res.get("clean", {"Rank-1": 0.0, "mAP": 0.0})
        self.logger.info(f"--> Validation: Rank-1: {clean_res['Rank-1']:.2f}% | mAP: {clean_res['mAP']:.2f}%")
        return clean_res

    def run(self):
        """
        Executes the full training schedule from start_epoch to total_epochs.
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING DG-ReID MULTI-STAGE TRAINING PIPELINE")
        env_info = get_environment_info()
        self.logger.info(f"Environment: Git Commit={env_info['git_commit']} | Device={env_info['device_name']}")
        self.logger.info("=" * 70)

        prev_stage = 0

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            stage = self.stage_manager.configure_model_for_epoch(self.model, epoch)

            # Rebuild optimizer parameter groups on stage transition
            if stage != prev_stage:
                summary = self.stage_manager.get_stage_summary(epoch)
                self.logger.info("\n" + "=" * 60)
                self.logger.info(f"ENTERED {summary['stage'].upper()}")
                self.logger.info(f"Trainable modules: {summary['trainable']}")
                self.logger.info(f"Frozen modules:    {summary['frozen']}")
                self.logger.info(f"Domain erasure:    {summary['adversarial']}")
                self.logger.info("=" * 60 + "\n")
                self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()
                prev_stage = stage

            # Run training epoch
            losses = self.train_epoch(epoch, stage)

            # Evaluate periodically or at stage completion
            val_metrics = {"Rank-1": 0.0, "mAP": 0.0}
            is_best = False

            if (epoch % self.eval_period == 0) or (epoch == self.total_epochs) or (epoch in (10, 30)):
                val_metrics = self.evaluate_model()
                rank1 = val_metrics.get("Rank-1", 0.0)
                map_val = val_metrics.get("mAP", 0.0)

                if rank1 > self.best_rank1:
                    self.best_rank1 = rank1
                    self.best_map = map_val
                    is_best = True
                    self.logger.info(f"[*] New best model found! Rank-1: {self.best_rank1:.2f}% | mAP: {self.best_map:.2f}%")

            # Save checkpoints
            checkpoint_state = {
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_rank1': self.best_rank1,
                'best_map': self.best_map,
                'config': self.config
            }

            save_checkpoint(checkpoint_state, self.checkpoint_dir, filename="checkpoint_latest.pth", is_best=is_best)

            if epoch in (10, 30, self.total_epochs):
                save_checkpoint(checkpoint_state, self.checkpoint_dir, filename=f"stage_{stage}_final.pth")

            # Record CSV row
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, stage, self.optimizer.param_groups[0]['lr'],
                    losses.get('loss_total', 0.0),
                    losses.get('loss_id_g', 0.0),
                    losses.get('loss_tri_g', 0.0),
                    losses.get('loss_id_l', 0.0),
                    losses.get('loss_tri_l', 0.0),
                    losses.get('loss_occ', 0.0),
                    losses.get('loss_dom', 0.0),
                    losses.get('loss_ortho', 0.0),
                    val_metrics.get('Rank-1', 0.0),
                    val_metrics.get('mAP', 0.0)
                ])
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"TRAINING COMPLETE. Best Rank-1: {self.best_rank1:.2f}%, Best mAP: {self.best_map:.2f}%")
        self.logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")
        self.logger.info("=" * 70)
