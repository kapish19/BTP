"""
Checkpointing and Model Selection for DG-ReID.
Saves model checkpoints, tracks best validation mAP / Rank-1,
and supports automatic resumption.
"""

import os
import torch


class CheckpointManager:
    def __init__(self, save_dir, logger=None):
        self.save_dir = save_dir
        self.logger = logger
        os.makedirs(save_dir, exist_ok=True)
        self.best_mAP = 0.0
        self.best_r1 = 0.0
        self.best_epoch = 0

    def save(self, epoch, model, optimizer=None, scheduler=None, metrics=None, is_best=False, filename=None):
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"

        save_path = os.path.join(self.save_dir, filename)
        raw_model = model.module if hasattr(model, 'module') else model

        # Extract state dict (optionally trainable only for lightweight storage)
        if getattr(self, 'save_trainable_only', False):
            # Save only heads, GAT, domain factors, and last 2 blocks
            trained_keys = ['global_proj', 'global_bn', 'global_classifier', 'part_gat',
                            'local_classifier', 'domain_classifier', 'domain_factors', 'resblocks.10', 'resblocks.11', 'ln_post']
            m_state = {k: v for k, v in raw_model.state_dict().items() if any(tk in k for tk in trained_keys)}
        else:
            m_state = raw_model.state_dict()

        state = {
            'epoch': epoch,
            'model_state_dict': m_state,
            'best_mAP': self.best_mAP,
            'best_r1': self.best_r1,
            'best_epoch': self.best_epoch
        }

        if optimizer is not None and getattr(self, 'save_optimizer', True):
            state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        if metrics is not None:
            state['metrics'] = metrics

        import shutil
        try:
            torch.save(state, save_path)
            if self.logger:
                self.logger.info(f"💾 Checkpoint saved: {save_path}")

            latest_path = os.path.join(self.save_dir, "checkpoint_latest.pth")
            shutil.copyfile(save_path, latest_path)

            if is_best:
                best_path = os.path.join(self.save_dir, "model_best.pth")
                shutil.copyfile(save_path, best_path)
                if self.logger:
                    self.logger.info(f"🏆 New best model saved! mAP: {self.best_mAP:.2%}, Rank-1: {self.best_r1:.2%}")
        except (RuntimeError, OSError) as e:
            # Automatic fallback: if disk is low, save only trained parameters
            if self.logger:
                self.logger.warning(f"⚠️ Full save failed ({e}). Attempting lightweight fallback...")
            trained_keys = ['global_proj', 'global_bn', 'global_classifier', 'part_gat',
                            'local_classifier', 'domain_classifier', 'domain_factors', 'resblocks.10', 'resblocks.11', 'ln_post']
            state['model_state_dict'] = {k: v for k, v in raw_model.state_dict().items() if any(tk in k for tk in trained_keys)}
            state.pop('optimizer_state_dict', None)
            torch.save(state, save_path)
            latest_path = os.path.join(self.save_dir, "checkpoint_latest.pth")
            shutil.copyfile(save_path, latest_path)
            if is_best:
                best_path = os.path.join(self.save_dir, "model_best.pth")
                shutil.copyfile(save_path, best_path)
            if self.logger:
                self.logger.info(f"💾 Saved lightweight trained parameters checkpoint: {save_path}")

    def update_best(self, mAP, r1, epoch):
        is_best = (mAP > self.best_mAP)
        if is_best:
            self.best_mAP = mAP
            self.best_r1 = r1
            self.best_epoch = epoch
        return is_best

    def resume(self, checkpoint_path, model, optimizer=None, scheduler=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        raw_model = model.module if hasattr(model, 'module') else model
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
        raw_model.load_state_dict(cleaned, strict=False)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_mAP = checkpoint.get('best_mAP', 0.0)
        self.best_r1 = checkpoint.get('best_r1', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)

        if self.logger:
            self.logger.info(f"🔄 Resumed checkpoint from epoch {start_epoch-1} ({checkpoint_path})")

        return start_epoch
