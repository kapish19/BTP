"""
Inference and Retrieval Utility for DG-ReID.
Supports:
  1. Extracting normalized 512-d embeddings for any person image crop.
  2. Ranking a gallery of images against a single query image.
"""

import argparse
import os
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.models.dg_reid import DGReID
from src.data.transforms import build_transforms
from src.utils.checkpoint import load_checkpoint


def extract_single_feature(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    transform=None
) -> np.ndarray:
    """
    Extracts L2-normalized 512-d retrieval embedding for a single image crop.
    """
    if transform is None:
        transform = build_transforms(img_size=(256, 128), is_train=False)

    with open(img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        feat = model(tensor)  # (1, 512)
    return feat.squeeze(0).cpu().numpy()


def rank_gallery_against_query(
    model: torch.nn.Module,
    query_path: str,
    gallery_dir: str,
    device: torch.device,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Ranks images in gallery_dir by cosine similarity to query image.
    """
    transform = build_transforms(img_size=(256, 128), is_train=False)
    q_feat = extract_single_feature(model, query_path, device, transform)

    gallery_files = [
        f for f in os.listdir(gallery_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not gallery_files:
        print(f"No valid images found in gallery directory: {gallery_dir}")
        return []

    scores = []
    for g_file in gallery_files:
        g_path = os.path.join(gallery_dir, g_file)
        g_feat = extract_single_feature(model, g_path, device, transform)
        # Cosine similarity: q_feat . g_feat (both L2 normalized)
        cos_sim = float(np.dot(q_feat, g_feat))
        scores.append((g_file, cos_sim))

    # Sort descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def main():
    parser = argparse.ArgumentParser(description="DG-ReID Inference Script")
    parser.add_argument("--query", type=str, required=True, help="Path to query image")
    parser.add_argument("--gallery", type=str, default=None, help="Optional gallery directory to rank")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--top_k", type=int, default=5, help="Top K matches to display")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")

    # Load model
    model = DGReID(num_classes=1000, num_domains=3).to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"[+] Loading checkpoint: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model, device=device, strict=False)
    else:
        print("[!] No checkpoint specified; running inference with initialized model.")

    if args.gallery:
        print(f"\n[*] Matching Query '{args.query}' against Gallery in '{args.gallery}'...")
        results = rank_gallery_against_query(model, args.query, args.gallery, device, top_k=args.top_k)
        print("\n" + "=" * 50)
        print(f"{'Rank':<6} | {'Gallery Filename':<28} | {'Cosine Sim':<10}")
        print("=" * 50)
        for rank, (fname, sim) in enumerate(results, 1):
            print(f"{rank:<6} | {fname:<28} | {sim:<10.4f}")
        print("=" * 50)
    else:
        feat = extract_single_feature(model, args.query, device)
        print(f"[+] Extracted 512-d normalized embedding for '{args.query}':")
        print(f"    Shape: {feat.shape} | L2 Norm: {np.linalg.norm(feat):.4f}")
        print(f"    First 10 dimensions: {feat[:10]}")


if __name__ == "__main__":
    main()
