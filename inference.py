"""
DG-ReID Inference Engine.
Loads a trained checkpoint and performs:
  1. Feature extraction for single or batch pedestrian images.
  2. Query-to-Gallery retrieval with cosine similarity or Euclidean distance.
  3. Top-K ranking and similarity score reporting.
  4. Robustness testing under arbitrary or synthetic query occlusion.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import DGReID
from datasets.transforms import build_transforms
from datasets.occlusion import apply_target_occlusion


def load_model(checkpoint_path, device, num_classes=1502, num_domains=3, backbone="ViT-B-16"):
    model = DGReID(
        num_classes=num_classes,
        num_domains=num_domains,
        backbone_name=backbone,
        img_size=(256, 128),
        feat_dim=512,
        pretrained=False
    )
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location='cpu')
        state_dict = state.get('model_state_dict', state)
        cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Running with initialized weights.")

    model.to(device)
    model.eval()
    return model


def extract_feature(model, img_path, transform, device, eval_feat="combined", occlusion="clean"):
    """
    Extracts L2-normalized 512-dimensional embedding for an image.
    """
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img) # [3, 256, 128]

    if occlusion != "clean":
        tensor = apply_target_occlusion(tensor, severity=occlusion)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(tensor, eval_feat=eval_feat) # [1, 512]
        feat = F.normalize(feat, p=2, dim=-1)

    return feat.squeeze(0).cpu()


def match_query_to_gallery(model, query_path, gallery_paths, transform, device, top_k=5, metric="cosine", occlusion="clean"):
    print(f"\nExtracting Query Feature: {query_path} (Occlusion: {occlusion})...")
    q_feat = extract_feature(model, query_path, transform, device, occlusion=occlusion)

    print(f"Extracting Features for {len(gallery_paths)} Gallery Images...")
    g_feats = []
    for g_path in gallery_paths:
        gf = extract_feature(model, g_path, transform, device, occlusion="clean")
        g_feats.append(gf)

    g_feats = torch.stack(g_feats, dim=0) # [N, 512]

    if metric == "cosine":
        # Cosine similarity (higher is better)
        sims = torch.mv(g_feats, q_feat) # [N]
        ranked_scores, ranked_idx = torch.sort(sims, descending=True)
    else:
        # Euclidean distance (lower is better)
        dists = torch.norm(g_feats - q_feat.unsqueeze(0), p=2, dim=1) # [N]
        ranked_scores, ranked_idx = torch.sort(dists, descending=False)

    print(f"\n================ TOP-{top_k} RETRIEVAL RESULTS ================")
    top_matches = []
    for rank in range(min(top_k, len(gallery_paths))):
        idx = ranked_idx[rank].item()
        score = ranked_scores[rank].item()
        path = gallery_paths[idx]
        top_matches.append((path, score))
        metric_str = f"Similarity: {score:.4f}" if metric == "cosine" else f"Distance: {score:.4f}"
        print(f"Rank {rank+1:02d}: {os.path.basename(path)} | {metric_str} | Path: {path}")

    return top_matches


def main():
    parser = argparse.ArgumentParser(description="DG-ReID Pedestrian Image Inference")
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth')
    parser.add_argument('--query', type=str, required=True, help="Path to query image")
    parser.add_argument('--gallery_dir', type=str, default='', help="Directory of gallery images")
    parser.add_argument('--gallery_imgs', nargs='+', default=[], help="List of gallery image paths")
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--occlusion', type=str, default='clean', choices=['clean', 'mild', 'moderate', 'severe'])
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) or args.device == 'cuda'
                          else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                          else 'cpu')

    model = load_model(args.checkpoint, device)
    transform = build_transforms(img_size=(256, 128), is_train=False)

    gallery = list(args.gallery_imgs)
    if args.gallery_dir and os.path.isdir(args.gallery_dir):
        for f in sorted(os.listdir(args.gallery_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                gallery.append(os.path.join(args.gallery_dir, f))

    if not gallery:
        print(f"Single image mode. Extracting feature for: {args.query}")
        feat = extract_feature(model, args.query, transform, device, occlusion=args.occlusion)
        print(f"Extracted feature tensor of shape: {feat.shape} (L2 Norm: {feat.norm(p=2).item():.4f})")
    else:
        match_query_to_gallery(
            model,
            args.query,
            gallery,
            transform,
            device,
            top_k=args.top_k,
            metric=args.metric,
            occlusion=args.occlusion
        )


if __name__ == "__main__":
    main()
