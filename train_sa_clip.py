"""
Training and Evaluation Script for SA-CLIP (Style-Agnostic CLIP for DG-ReID).
Trains on source domain (e.g. Market-1501) and evaluates zero-shot on unseen target domain (e.g. DukeMTMC-reID).
"""

import os
import re
import glob
import time
import math
import random
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms

from models.sa_clip import SACLIP


# ==============================================================================
# 1. Dataset & Data Loading
# ==============================================================================

class ReIDDataset(Dataset):
    """Standard Re-ID Dataset parser for Market-1501 and DukeMTMC-reID."""
    def __init__(self, data_list, transform=None, is_train=False):
        self.data_list = data_list  # List of (img_path, pid, camid)
        self.transform = transform
        self.is_train = is_train

        # Relabel PIDs to 0..C-1 for training
        if self.is_train:
            pids = sorted(list(set([x[1] for x in self.data_list])))
            self.pid2label = {pid: label for label, pid in enumerate(pids)}
            self.num_classes = len(pids)
        else:
            self.num_classes = len(set([x[1] for x in self.data_list]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path, pid, camid = self.data_list[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.pid2label[pid] if self.is_train else pid
        return img, label, camid, img_path


def parse_reid_dir(dir_path):
    """Parses images from standard Re-ID folder (bounding_box_train, query, bounding_box_test)."""
    pattern = re.compile(r'([-\d]+)_c(\d+)')
    img_paths = sorted(glob.glob(os.path.join(dir_path, '*.jpg')) + glob.glob(os.path.join(dir_path, '*.png')))
    data = []
    for path in img_paths:
        fname = os.path.basename(path)
        match = pattern.search(fname)
        if not match:
            continue
        pid, camid = match.groups()
        pid, camid = int(pid), int(camid)
        if pid == -1:
            continue  # Skip junk images
        data.append((path, pid, camid))
    return data


class RandomIdentitySampler(Sampler):
    """PK Sampler: Randomly samples P identities and K instances per identity for each batch."""
    def __init__(self, data_list, batch_size=64, num_instances=4):
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        self.index_dict = defaultdict(list)
        for idx, (_, pid, _) in enumerate(self.data_list):
            self.index_dict[pid].append(idx)
        self.pids = list(self.index_dict.keys())

    def __iter__(self):
        batch_idxs_dict = {}
        for pid in self.pids:
            idxs = list(self.index_dict[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)
            batch_idxs_dict[pid] = [
                idxs[i:i + self.num_instances] for i in range(0, len(idxs) - self.num_instances + 1, self.num_instances)
            ]

        avai_pids = list(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return (len(self.pids) // self.num_pids_per_batch) * self.batch_size


# ==============================================================================
# 2. Loss Functions
# ==============================================================================

class CrossEntropyLabelSmooth(nn.Module):
    """Cross-entropy loss with label smoothing to prevent overfitting."""
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        return (- targets_smooth * log_probs).sum(dim=1).mean()


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


class HardTripletLoss(nn.Module):
    """Batch-hard triplet loss for metric clustering."""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        dist_mat = euclidean_dist(feats, feats)
        N = dist_mat.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        dist_ap = torch.max(dist_mat[is_pos].view(N, -1), 1)[0]
        dist_an = torch.min(dist_mat[is_neg].view(N, -1), 1)[0]

        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


# ==============================================================================
# 3. Evaluation Metrics (Official Market-1501 protocol)
# ==============================================================================

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        # Discard gallery images of the same person from the same camera
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)

    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def evaluate(model, query_loader, gallery_loader, device):
    """Extracts L2-normalized features and computes mAP and CMC Rank-1/5/10."""
    model.eval()
    q_feats, q_pids, q_camids = [], [], []
    g_feats, g_pids, g_camids = [], [], []

    with torch.no_grad():
        for imgs, pids, camids, _ in query_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            q_feats.append(feats.cpu())
            q_pids.extend(pids)
            q_camids.extend(camids)

        for imgs, pids, camids, _ in gallery_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            g_feats.append(feats.cpu())
            g_pids.extend(pids)
            g_camids.extend(camids)

    q_feats = torch.cat(q_feats, dim=0)
    g_feats = torch.cat(g_feats, dim=0)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    # Compute Euclidean distance on L2-normalized features
    m, n = q_feats.size(0), g_feats.size(0)
    distmat = torch.pow(q_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(g_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(q_feats, g_feats.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    return cmc, mAP


# ==============================================================================
# 4. Training Engine
# ==============================================================================

def train_sa_clip(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build transforms
    train_transforms = [
        transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if args.use_rea:
        train_transforms.append(transforms.RandomErasing(p=0.5, value='random'))
    train_transform = transforms.Compose(train_transforms)

    val_transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Find dataset directories
    source_train_dir = os.path.join(args.data_path, args.source_dataset, 'bounding_box_train')
    if not os.path.exists(source_train_dir):
        # Look for nested folder
        cands = glob.glob(os.path.join(args.data_path, '**', 'bounding_box_train'), recursive=True)
        if cands:
            source_train_dir = cands[0]

    target_query_dir = os.path.join(args.data_path, args.target_dataset, 'query')
    target_gallery_dir = os.path.join(args.data_path, args.target_dataset, 'bounding_box_test')
    if not os.path.exists(target_query_dir):
        qcands = glob.glob(os.path.join(args.data_path, '**', 'query'), recursive=True)
        gcands = glob.glob(os.path.join(args.data_path, '**', 'bounding_box_test'), recursive=True)
        if len(qcands) > 1 and len(gcands) > 1:
            target_query_dir = qcands[1]
            target_gallery_dir = gcands[1]
        elif qcands and gcands:
            target_query_dir = qcands[0]
            target_gallery_dir = gcands[0]

    print(f"Source train dir: {source_train_dir}")
    print(f"Target query dir: {target_query_dir}")
    print(f"Target gallery dir: {target_gallery_dir}")

    # Build datasets
    train_data = parse_reid_dir(source_train_dir)
    query_data = parse_reid_dir(target_query_dir)
    gallery_data = parse_reid_dir(target_gallery_dir)

    print(f"Source Training Samples: {len(train_data)} images")
    print(f"Target Query Samples: {len(query_data)} images")
    print(f"Target Gallery Samples: {len(gallery_data)} images")

    train_set = ReIDDataset(train_data, transform=train_transform, is_train=True)
    query_set = ReIDDataset(query_data, transform=val_transform, is_train=False)
    gallery_set = ReIDDataset(gallery_data, transform=val_transform, is_train=False)

    num_classes = train_set.num_classes
    print(f"Number of training identities: {num_classes}")

    train_sampler = RandomIdentitySampler(train_data, batch_size=args.batch_size, num_instances=4)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build SA-CLIP Model
    model = SACLIP(
        num_classes=num_classes,
        backbone=args.backbone,
        img_size=(256, 128),
        use_mixstyle=args.use_mixstyle,
        mixstyle_p=args.mixstyle_p,
        mixstyle_alpha=args.mixstyle_alpha
    ).to(device)

    # Optimizer: separate backbone lr and head lr
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bottleneck' in name or 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone, 'weight_decay': args.weight_decay},
        {'params': head_params, 'lr': args.lr_head, 'weight_decay': args.weight_decay}
    ])

    # Warmup + Cosine Annealing Scheduler
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5.0
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Losses
    criterion_ce = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1)
    criterion_tri = HardTripletLoss(margin=0.3)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_rank1 = 0.0

    print("\n" + "=" * 60)
    print(f"Starting SA-CLIP Training ({args.epochs} Epochs)")
    print(f"MixStyle Enabled: {args.use_mixstyle} | Random Erasing: {args.use_rea}")
    print("=" * 60 + "\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_ce, total_tri = 0.0, 0.0, 0.0
        start_time = time.time()

        for step, (imgs, pids, _, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            pids = pids.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                cls_score, global_feat = model(imgs)
                loss_ce = criterion_ce(cls_score, pids)
                loss_tri = criterion_tri(global_feat, pids)
                loss = loss_ce + loss_tri

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_tri += loss_tri.item()

        scheduler.step()
        elapsed = time.time() - start_time
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_ce = total_ce / num_batches
        avg_tri = total_tri / num_batches

        b_lr = optimizer.param_groups[0]['lr']
        h_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) - Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Tri: {avg_tri:.4f}) | Backbone LR: {b_lr:.1e}, Head LR: {h_lr:.1e}")

        # Evaluation
        if epoch % args.eval_period == 0 or epoch == args.epochs:
            print(f"\n--- Zero-Shot Evaluation on Target Domain: {args.target_dataset} ---")
            cmc, mAP = evaluate(model, query_loader, gallery_loader, device)
            print(f"mAP: {mAP:.1%} | Rank-1: {cmc[0]:.1%} | Rank-5: {cmc[4]:.1%} | Rank-10: {cmc[9]:.1%}")

            if cmc[0] > best_rank1:
                best_rank1 = cmc[0]
                save_path = os.path.join(args.checkpoint_dir, 'sa_clip_best.pth')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'rank1': cmc[0], 'mAP': mAP}, save_path)
                print(f"💾 Saved New Best Checkpoint (Rank-1: {cmc[0]:.1%}) to {save_path}")
            print("-" * 60 + "\n")

    print(f"\nTraining Complete! Best Unseen Target Rank-1: {best_rank1:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SA-CLIP DG-ReID Runner")
    parser.add_argument('--source_dataset', type=str, default='Market-1501-v15.09.15')
    parser.add_argument('--target_dataset', type=str, default='DukeMTMC-reID')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval_period', type=int, default=5)
    parser.add_argument('--lr_backbone', type=float, default=5e-6)
    parser.add_argument('--lr_head', type=float, default=3.5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--use_mixstyle', action='store_true', default=True)
    parser.add_argument('--no_mixstyle', dest='use_mixstyle', action='store_false')
    parser.add_argument('--mixstyle_p', type=float, default=0.5)
    parser.add_argument('--mixstyle_alpha', type=float, default=0.1)
    parser.add_argument('--use_rea', action='store_true', default=True)
    parser.add_argument('--no_rea', dest='use_rea', action='store_false')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_sa_clip')

    args = parser.parse_args()
    train_sa_clip(args)
