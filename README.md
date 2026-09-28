# Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification (DG-ReID)

[![CI](https://github.com/kapish19/btp-/actions/workflows/ci.yml/badge.svg)](https://github.com/kapish19/btp-/actions/workflows/ci.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official research-grade PyTorch implementation of the B.Tech Project (BTP-I) report:  
**"Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification"**  
*Department of Computer Science and Engineering, Netaji Subhas University of Technology (NSUT), New Delhi.*

**Authors:** Diya Bangera (2023UCA1917), Kapish Verma (2023UCS1632), Rishit Rana (2023UCA1911)  
**Supervisor:** Dr. Vandana Bhatia (Assistant Professor, Department of CSE, NSUT)

---

## 📖 Table of Contents
- [1. Research Overview & Motivation](#1-research-overview--motivation)
- [2. Architectural Highlights & Contributions](#2-architectural-highlights--contributions)
- [3. Mathematical Formulation](#3-mathematical-formulation)
- [4. Repository Structure](#4-repository-structure)
- [5. Installation & Environment Setup](#5-installation--environment-setup)
- [6. Verification & Smoke Testing (Section 6.7)](#6-verification--smoke-testing-section-67)
- [7. Dataset Preparation](#7-dataset-preparation)
- [8. Training Pipeline (3-Stage Schedule)](#8-training-pipeline-3-stage-schedule)
- [9. Evaluation & Ablation Studies (Tables 7 & 8)](#9-evaluation--ablation-studies-tables-7--8)
- [10. Standalone Inference](#10-standalone-inference)
- [11. Experimental Results](#11-experimental-results)
- [12. GitHub Setup & Push Commands](#12-github-setup--push-commands)
- [13. License & Citation](#13-license--citation)

---

## 1. Research Overview & Motivation

In open-world deployments, Person Re-Identification (Re-ID) encounters two simultaneous, compounding challenges:
1. **Domain Shift (Distribution Mismatch):** Variations in camera hardware, lighting, viewpoints, scene backgrounds, and color distributions cause deep neural networks to learn non-causal shortcuts instead of identity-intrinsic representations.
2. **Partial Occlusion:** Obstacles such as vehicles, bicycles, pillars, or other pedestrians block critical body parts, corrupting global features and misleading fixed pooling mechanisms.

Under heavy occlusion, models that lose informative body features rely even more heavily on domain-specific nuisance factors (background, camera color style). 

### How DG-ReID Differs from Base Paper (CILP-FGDI)
| Property | CILP-FGDI (Base Paper) [13] | **DG-ReID (Proposed Framework)** |
| :--- | :--- | :--- |
| **Domain Erasure Mechanism** | Text-side adversarial prompt tuning | **Direct Visual-side Gradient Reversal Layer (GRL)** on the class token $f_{cls}$ |
| **Inference Dependency** | Requires text encoder during retrieval | **Pure visual-only forward path** (zero text-side overhead at deployment) |
| **Occlusion Handling** | Implicit global attention | **Visibility-Aware Part GAT** with explicit synthetic mask supervision ($L_{occ}$) |
| **Local Modeling** | Flat token aggregation | **16×8 spatial patch grid** mapped to 3 coarse vertical body parts ($R_1, R_2, R_3$) |
| **Feature Fusion** | Text-guided image features | Multi-granular normalized fusion: $f = \frac{g + l}{\|g + l\|_2} \in \mathbb{R}^{512}$ |

---

## 2. Architectural Highlights & Contributions

![DG-ReID Architecture](f1.png)

The framework follows a **shared-backbone, dual-evidence** design:
- **Visual Backbone:** CLIP ViT-B/16 visual transformer with bicubic interpolation of positional embeddings to accept $256 \times 128$ resolution, yielding a $16 \times 8$ spatial token grid (128 patch tokens + 1 class token).
- **Global Identity Branch:** Class token $f_{cls} \in \mathbb{R}^{768}$ projected via $W_g$ to 512 dimensions, followed by BNNeck: $g = \text{BN}(h_g)$. Supervised with label-smoothed identity classification ($L_{id}^g$) and batch-hard triplet loss ($L_{tri}^g$, margin $m=0.3$).
- **Domain-Adversarial Global Branch:** Class token $f_{cls}$ passes through a Gradient Reversal Layer ($R_{\alpha(e)}$) into an MLP domain classifier ($768 \to 384 \to M$), forcing the backbone to purge domain-specific shortcuts.
- **Part Construction & Pooling:** Patch tokens $F_{patch} \in \mathbb{R}^{128 \times 768}$ partitioned on the $16 \times 8$ grid into 3 coarse regions:
  - $R_1$ (Upper body / Head / Shoulders): rows 0..4 (40 patches)
  - $R_2$ (Middle body / Torso): rows 5..10 (48 patches)
  - $R_3$ (Lower body / Legs): rows 11..15 (40 patches)
- **PartVisibilityGAT:** 3-node Graph Attention Network models inter-node context exchange:
  $$e_{jk} = \frac{(W_Q p_j)^\top (W_K p_k)}{\sqrt{d_a}}, \quad a_{jk} = \text{softmax}(e_{jk}), \quad p'_j = \sum_{k=1}^3 a_{jk} W_V p_k$$
- **Internal Reconstruction Gate:** Blends uncorrupted evidence with graph context: $\tilde{p}_j = \gamma_j p'_j + (1 - \gamma_j) p_j$, where $\gamma_j = \sigma(w_r^\top p_j + b_r)$.
- **Synthetic Occlusion & Visibility Prediction:** Supervised by continuous visibility targets $m_{ij} \in [0, 1]$ derived from exact patch-grid occlusion masks: $v_{ij} = \sigma(w_2^\top \delta(W_1 p'_{ij} + b_1) + b_2)$.
- **Local Representation:** Modulated parts $\hat{p}_j = v_{ij} \tilde{p}_j$ aggregated and projected: $h_l = W_l (\sum_{j=1}^3 \hat{p}_j) + b_l \to \text{BN} \to l \in \mathbb{R}^{512}$.
- **Orthogonal Domain Factors:** $K=4$ learnable vectors $G \in \mathbb{R}^{512 \times K}$ constrained by $L_{ortho} = \|\hat{G}^\top \hat{G} - I_K\|_F^2$.
- **Retrieval Fusion:** $f = \frac{g + l}{\|g + l\|_2} \in \mathbb{R}^{512}$.

---

## 3. Mathematical Formulation

### Complete Joint Optimization Objective (Eq. 1)
$$\mathcal{L}_{total} = \lambda_{id} L_{id}^g + \lambda_{tri} L_{tri}^g + \lambda_{id}^{loc} L_{id}^l + \lambda_{tri}^{loc} L_{tri}^l + \lambda_{occ} L_{occ} + \lambda_{adv} L_{dom} + \lambda_{ortho} L_{ortho}$$

**Design Defaults (Table 2):**
| Loss Term | Notation | Weight | Purpose |
| :--- | :---: | :---: | :--- |
| **Global Identity Loss** | $L_{id}^g$ | $1.0$ | Holistic identity separation (Label smoothing $\epsilon=0.1$) |
| **Global Triplet Loss** | $L_{tri}^g$ | $1.0$ | Hard positive/negative metric clustering (Margin $m=0.3$) |
| **Local Identity Loss** | $L_{id}^l$ | $0.5$ | Part-level identity discriminability |
| **Local Triplet Loss** | $L_{tri}^l$ | $0.5$ | Local metric margin enforcement |
| **Visibility Loss** | $L_{occ}$ | $0.3$ | Soft BCE on continuous visibility ratio $m_{ij} \in [0, 1]$ |
| **Domain Adversarial Loss** | $L_{dom}$ | $0.1$ | Source domain confusion via GRL |
| **Orthogonality Loss** | $L_{ortho}$ | $0.05$ | Latent domain factor diversity: $\|\hat{G}^\top \hat{G} - I_K\|_F^2$ |

### Dynamic GRL Warmup Schedule (Section 5.5)
$$\alpha(e) = \min\left(1.0, \frac{e - 30}{10}\right), \quad e \ge 31$$
During Stages 1 & 2 ($e \le 30$), $\alpha(e) = 0.0$ to stabilize representations before domain confusion.

---

## 4. Repository Structure

```
├── cfgs/
│   ├── cfgs.py                 # CLI argument definitions & protocol configurations
│   └── yaml_configs/           # YAML configs for full model and ablations
│       ├── full_dgreid.yaml    # Full proposed DG-ReID
│       ├── baseline_clip.yaml  # CILP-FGDI visual baseline
│       ├── grl_only.yaml       # GRL-Only ablation
│       ├── part_only.yaml      # Part-Only ablation
│       └── occluded_eval.yaml  # Occlusion robustness benchmarks
├── datasets/
│   ├── bases.py                # Base dataset abstractions
│   ├── build.py                # DataLoader builder with PK sampler
│   ├── common.py               # CommDataset with integrated synthetic occlusion
│   ├── occlusion.py            # Synthetic occlusion transform & target generator
│   ├── transforms.py           # Standard Re-ID transforms (256x128)
│   ├── market1501.py           # Market-1501 benchmark parser
│   ├── msmt17.py               # MSMT17 benchmark parser
│   ├── cuhk03.py               # CUHK03 benchmark parser
│   ├── cuhk_sysu.py            # CUHK-SYSU benchmark parser
│   ├── occluded_duke.py        # Occluded-DukeMTMC benchmark parser
│   ├── samplers/               # PK Sampler (P=16, K=4, B=64)
│   ├── prepare_datasets.py     # Automated dataset downloader (Kaggle & Drive)
│   └── create_dummy_data.py    # Synthetic dataset generator for tests
├── models/
│   ├── clip/                   # OpenAI CLIP ViT-B/16 implementation
│   ├── grl.py                  # Gradient Reversal Layer & dynamic schedule
│   ├── domain_classifier.py    # MLP Domain Classifier (768 -> 384 -> M)
│   ├── part_gat.py             # PartVisibilityGAT, reconstruction gate, visibility head
│   ├── domain_factors.py       # OrthogonalDomainFactors (K=4 vectors)
│   ├── baseline.py             # Baseline CLIP model wrapper
│   └── dgreid_model.py         # Complete unified DGReID architecture
├── loss/
│   ├── make_loss.py            # TotalLoss implementing Eq. 1 with stage gating
│   ├── softmax_loss.py         # Label-smoothed cross-entropy loss
│   └── triplet_loss.py         # Batch-hard triplet loss (margin 0.3)
├── solver/
│   ├── make_optimizer.py       # Dual-rate optimizer (backbone lr vs head lr)
│   ├── lr_scheduler.py         # Warmup cosine & multi-step schedulers
│   └── stage_manager.py        # 3-Stage parameter freezing manager (Table 3)
├── reidutils/
│   ├── checkpoint.py           # Checkpointing, resume, and best-model tracking
│   ├── metrics.py              # Market-1501 CMC (Rank-1/5/10) and mAP evaluator
│   └── logger.py               # Formatted logging utilities
├── experiments/
│   ├── run_ablations.py        # Automated runner for Table 7 & Table 8 experiments
│   └── generate_tables.py      # Markdown & LaTeX table generator
├── tests/
│   ├── test_smoke.py           # 8 mandatory smoke tests (Section 6.7 of report)
│   └── test_pipeline.py        # End-to-end 3-stage mini pipeline test on synthetic data
├── inference.py                # Standalone inference on arbitrary pedestrian images
├── run.py                      # Main entrypoint for training and evaluation
├── setup.py                    # Package installer
├── requirements.txt            # Dependency specification
├── environment.yml             # Conda environment specification
└── .github/workflows/ci.yml    # GitHub Actions Continuous Integration
```

---

## 5. Installation & Environment Setup

### Option A: Using Pip
```bash
git clone https://github.com/kapish19/btp-.git
cd btp-
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option B: Using Conda
```bash
conda env create -f environment.yml
conda activate dgreid
```

---

## 6. Verification & Smoke Testing (Section 6.7)

Before launching full-scale training, execute the mandatory verification suite to validate all 8 architectural checks from Section 6.7 of the report:

```bash
python tests/test_smoke.py
```

**Verified Checks:**
- [x] **Check 1:** CLIP ViT-B/16 output tensor shapes ($f_{cls} \in \mathbb{R}^{B \times 768}$, $F_{patch} \in \mathbb{R}^{B \times 128 \times 768}$) for $256 \times 128$.
- [x] **Check 2:** 16×8 patch grid reconstruction into 3 vertical parts ($R_1$: rows 0..4, $R_2$: rows 5..10, $R_3$: rows 11..15).
- [x] **Check 3:** Synthetic occlusion mask alignment with corrupted image patches.
- [x] **Check 4:** Visibility target bounds ($m_{ij} = 1.0$ when fully visible, $m_{ij} = 0.0$ when fully occluded).
- [x] **Check 5:** Non-zero gradient propagation to all heads (`global_proj`, `global_classifier`, `part_gat`, `local_classifier`, `vis_mlp`, `domain_factors`).
- [x] **Check 6:** Reversed gradient sign through GRL ($\frac{\partial R_\alpha}{\partial x} = -\alpha I$).
- [x] **Check 7:** Finite loss values across all 3 training stages.
- [x] **Check 8:** Checkpoint serialization and restoration numerical equivalence ($< 10^{-6}$).

### Run Full Synthetic Pipeline Integration Test
Verify the entire 3-stage training loop, evaluation, and checkpoint resumption without needing any real dataset:
```bash
python tests/test_pipeline.py
```

---

## 7. Dataset Preparation

DG-ReID supports the standard multi-source Person Re-ID datasets:
- **Market-1501:** 1,502 identities, 32,668 images, 6 cameras.
- **MSMT17 (V2):** 4,101 identities, 126,441 images, 15 cameras.
- **CUHK03 (Detected/Labeled):** 1,467 identities, 14,096 images, 10 cameras.
- **CUHK-SYSU:** 11,934 identities, cropped pedestrian crops.
- **Occluded-DukeMTMC:** Occlusion benchmark for zero-shot testing.

### Directory Organization
Organize dataset archives under `./data` as follows:
```
data/
├── Market-1501-v15.09.15/
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
├── MSMT17_V2/
│   ├── mask_train_v2/
│   ├── mask_test_v2/
│   ├── list_train.txt
│   └── list_val.txt
├── cuhk03/
│   └── cuhk03_release/
└── cuhk_sysu/
    └── cropped_images/
```

### Automated Dataset Setup
Download and unpack datasets using Kaggle API or Google Drive:
```bash
# Configure Kaggle API (ensure ~/.kaggle/kaggle.json exists)
python prepare_datasets.py --data_dir data
```

---

## 8. Training Pipeline (3-Stage Schedule)

Training executes across **3 stages** as specified in Table 3:
- **Stage 1 (Epochs 1–10):** Visual backbone is frozen. Stabilizes `global_proj`, BNNecks, classifiers, `PartVisibilityGAT`, visibility head, and domain factors (`head_lr = 3.5e-4`).
- **Stage 2 (Epochs 11–30):** Unfreezes the final two ViT transformer blocks (`image_encoder.transformer.resblocks[-2:]`) with `base_lr = 5e-6` to adapt high-level features.
- **Stage 3 (Epochs 31–60):** Activates domain classifier behind GRL with linear warmup: $\alpha(e) = \min(1, \frac{e-30}{10})$.

### Leave-One-Domain-Out Training Commands (Protocol 2)

#### 1. Hold out Market-1501 (Train on MSMT17 + CUHK-SYSU + CUHK03):
```bash
python run.py \
    --benchmark protocol2 \
    --held_out_domain Market \
    --data_path data \
    --batch_size 64 \
    --num_instance 4 \
    --epochs 60 \
    --stage1_epochs 10 \
    --stage2_epochs 30 \
    --head-lr 0.00035 \
    --image-encoder-lr 0.000005 \
    --device auto
```

#### 2. Hold out MSMT17 (Train on Market + CUHK-SYSU + CUHK03):
```bash
python run.py \
    --benchmark protocol2 \
    --held_out_domain MSMT17 \
    --data_path data \
    --batch_size 64 \
    --device auto
```

#### 3. Hold out CUHK03 (Train on Market + MSMT17 + CUHK-SYSU):
```bash
python run.py \
    --benchmark protocol2 \
    --held_out_domain cuhk03 \
    --data_path data \
    --batch_size 64 \
    --device auto
```

#### Resuming Interrupted Training
```bash
python run.py \
    --benchmark protocol2 \
    --held_out_domain Market \
    --resume logs/protocol_2_Market_ViT-B-16_.../checkpoint_latest.pth
```

---

## 9. Evaluation & Ablation Studies (Tables 7 & 8)

### Table 7: Component-Wise Ablations
Run individual ablation configurations:

```bash
# 1. CILP-FGDI Visual Baseline (No GRL, No Part branch)
python run.py --benchmark protocol2 --held_out_domain Market \
    --disable_grl --disable_part_branch --eval_feat global

# 2. GRL-Only (Yes GRL, No Part branch)
python run.py --benchmark protocol2 --held_out_domain Market \
    --disable_part_branch --eval_feat global

# 3. Part-Only (No GRL, Yes Part branch)
python run.py --benchmark protocol2 --held_out_domain Market \
    --disable_grl --eval_feat combined

# 4. Full DG-ReID (Yes GRL, Yes Part branch)
python run.py --benchmark protocol2 --held_out_domain Market \
    --eval_feat combined
```

Or run all Table 7 ablations automatically:
```bash
python experiments/run_ablations.py --held_out_domain Market --output_dir ablation_results
```

### Table 8: Target Query Occlusion Robustness
Evaluate model degradation under controlled synthetic target-query occlusions:
```bash
# Clean (0% occlusion)
python run.py --benchmark protocol2 --held_out_domain Market --eval_only \
    --resume logs/model_best.pth --occlusion_severity clean

# Mild (~15% occlusion)
python run.py --benchmark protocol2 --held_out_domain Market --eval_only \
    --resume logs/model_best.pth --occlusion_severity mild

# Moderate (~30% occlusion)
python run.py --benchmark protocol2 --held_out_domain Market --eval_only \
    --resume logs/model_best.pth --occlusion_severity moderate

# Severe (~50% occlusion)
python run.py --benchmark protocol2 --held_out_domain Market --eval_only \
    --resume logs/model_best.pth --occlusion_severity severe
```

### Automatically Generate Publication Tables (Markdown & LaTeX)
```bash
python experiments/generate_tables.py --results_dir ablation_results --save_md EXPERIMENT_RESULTS.md
```

---

## 10. Standalone Inference

Perform feature extraction or rank a gallery against an arbitrary query image:

```bash
python inference.py \
    --checkpoint logs/model_best.pth \
    --query path/to/query.jpg \
    --gallery_dir path/to/gallery/ \
    --top_k 5 \
    --metric cosine \
    --occlusion clean
```

---

## 11. Experimental Results

### Table 7: Component-Ablation Results (Rank-1 and mAP, %)
*Leave-one-domain-out evaluation on unseen Market-1501.*

| Configuration | Visual GRL | Part Branch | Rank-1 (%) | mAP (%) |
| :--- | :---: | :---: | :---: | :---: |
| **CILP-FGDI Baseline** | No | No | 80.1 | 57.1 |
| **GRL-Only** | Yes | No | 82.2 (+2.1) | 59.5 (+2.4) |
| **Part-Only** | No | Yes | 81.0 (+0.9) | 58.5 (+1.4) |
| **Full DG-ReID (Ours)** | **Yes** | **Yes** | **83.5 (+3.4)** | **61.2 (+4.1)** |

### Table 8: Rank-1 (%) Under Increasing Target Occlusion
*Evaluated across Clean, Mild, Moderate, and Severe target-query occlusions.*

| Configuration | Clean (0%) | Mild (15%) | Moderate (30%) | Severe (50%) | Degradation Curve |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Global Baseline** | 80.1% | 72.0% | 55.0% | 32.0% | -48.1% (Severe drop) |
| **+ Part Branch (No Vis)** | 80.5% | 74.5% | 61.0% | 41.0% | -39.5% |
| **Full DG-ReID (Ours)** | **83.5%** | **80.0%** | **71.5%** | **55.0%** | **-28.5% (Highest resilience)** |

---

## 12. GitHub Setup & Push Commands

To publish this codebase directly to GitHub, run the following commands:

```bash
# 1. Initialize and configure local branch
git checkout -b main || git checkout main

# 2. Stage all implementation modules, tests, and configurations
git add models/ datasets/ loss/ solver/ reidutils/ experiments/ tests/ cfgs/
git add run.py inference.py setup.py requirements.txt environment.yml README.md LICENSE .gitignore .github/

# 3. Create a clean reproducible commit
git commit -m "feat: complete research-grade DG-ReID implementation (PartVisibilityGAT, GRL, 3-stage training, and verification suite)"

# 4. Push to remote repository
git push -u origin main
```

---

## 13. License & Citation

This project is licensed under the [MIT License](LICENSE).

```bibtex
@article{bangera2026visibility,
  title={Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification},
  author={Bangera, Diya and Verma, Kapish and Rana, Rishit and Bhatia, Vandana},
  journal={B.Tech Project Report, Netaji Subhas University of Technology (NSUT)},
  year={2026}
}
```
