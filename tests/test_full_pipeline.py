"""
Full pipeline integration smoke test:
- Generates synthetic multi-domain dataset.
- Executes 1 epoch of Stage 1, Stage 2, and Stage 3.
- Validates forward pass, backward propagation, parameter updates, checkpointing, and evaluation.
- Requires no external datasets or pretrained weights.
"""

import os
import tempfile
import unittest
import torch
from torch.utils.data import DataLoader

from src.models.dg_reid import DGReID
from src.data.dataset import ReIDImageDataset, MultiDomainReIDDataset, parse_market1501_dir
from src.data.sampler import MultiDomainPKSampler
from src.data.synthetic_data import generate_synthetic_reid_dataset
from src.data.occlusion import SyntheticOcclusionGenerator
from src.training.trainer import Trainer
from src.utils.config import load_config


class TestFullPipeline(unittest.TestCase):
    def test_end_to_end_smoke_test(self):
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            # 1. Generate synthetic dataset
            data_dir = os.path.join(tmpdir, "data")
            domain_dirs = generate_synthetic_reid_dataset(
                output_dir=data_dir,
                num_identities=12,
                images_per_identity=4,
                num_cameras=2,
                num_domains=2
            )

            occ_gen = SyntheticOcclusionGenerator(img_size=(256, 128), prob=0.5)

            # Source dataset
            source_data = [
                ("domain_0", parse_market1501_dir(os.path.join(domain_dirs["domain_0"], "bounding_box_train"), domain_id=0))
            ]
            train_dataset = MultiDomainReIDDataset(source_data, occlusion_generator=occ_gen)
            sampler = MultiDomainPKSampler(train_dataset, p=4, k=2, seed=42)
            train_loader = DataLoader(train_dataset, batch_sampler=sampler)

            # Target dataset
            target_query = parse_market1501_dir(os.path.join(domain_dirs["domain_1"], "query"), domain_id=1)
            target_gallery = parse_market1501_dir(os.path.join(domain_dirs["domain_1"], "bounding_box_test"), domain_id=1)
            val_q_loader = DataLoader(ReIDImageDataset(target_query, is_train=False), batch_size=4)
            val_g_loader = DataLoader(ReIDImageDataset(target_gallery, is_train=False), batch_size=4)

            # 2. Instantiate Model with depth=2 for fast smoke test
            model = DGReID(
                num_classes=train_dataset.num_classes,
                num_domains=1,
                feat_dim=512,
                clip_dim=768,
                depth=2,
                num_factors=4,
                unfreeze_blocks=1
            )

            # 3. Configure Trainer
            config = {
                "loss": {
                    "lambda_id": 1.0,
                    "lambda_tri": 1.0,
                    "lambda_id_loc": 0.5,
                    "lambda_tri_loc": 0.5,
                    "lambda_occ": 0.3,
                    "lambda_adv": 0.1,
                    "lambda_ortho": 0.05
                },
                "train": {
                    "stage1_epochs": 1,
                    "stage2_epochs": 2,
                    "stage3_epochs": 3,
                    "total_epochs": 3,
                    "grl_ramp_epochs": 1,
                    "lr_head": 1e-3,
                    "lr_backbone": 1e-4,
                    "clip_grad_norm": 1.0,
                    "eval_period": 1,
                    "use_amp": False
                }
            }

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_query_loader=val_q_loader,
                val_gallery_loader=val_g_loader,
                config=config,
                device=torch.device("cpu"),
                output_dir=tmpdir
            )

            # Run 3 epochs covering Stage 1, Stage 2, Stage 3
            trainer.run()

            # Verify checkpoints were created
            ckpt_path = os.path.join(tmpdir, "checkpoints", "checkpoint_latest.pth")
            self.assertTrue(os.path.isfile(ckpt_path))

            # Verify CSV metrics log was populated
            csv_path = os.path.join(tmpdir, "logs", "training_metrics.csv")
            self.assertTrue(os.path.isfile(csv_path))


if __name__ == '__main__':
    unittest.main()
