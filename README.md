# Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/kapishverma/dg-reid/actions/workflows/ci.yml/badge.svg)](https://github.com/kapishverma/dg-reid/actions)

> **Official Research Codebase for B.Tech Project-I (BTP-I)**  
> **Authors:** Diya Bangera (2023UCA1917), Kapish Verma (2023UCS1632), Rishit Rana (2023UCA1911)  
> **Supervisor:** Dr. Vandana Bhatia, Assistant Professor, Department of Computer Science and Engineering  
> **Institution:** Netaji Subhas University of Technology (NSUT), New Delhi  

---

## 1. Overview & Motivation

Person Re-Identification (Re-ID) in real-world deployment faces two simultaneous challenges:
1. **Domain Shift:** Target surveillance networks possess different camera hardware, illumination, background scenes, and viewpoints compared to training domains.
2. **Partial Occlusion:** Pedestrians are frequently occluded by obstacles (vehicles, poles, foliage) or other pedestrians, corrupting spatial evidence.

When a body region (e.g., the torso) is occluded, conventional holistic models overfit to domain-specific nuisance shortcuts (background color, camera style, lighting). Existing methods like CILP-FGDI address domain generalization via text-side prompt tuning, requiring complex vision-language text pipelines during training.

### Key Contributions of DG-ReID:
- **Direct Visual-Side Domain Erasure:** Integrates a Gradient Reversal Layer (GRL) directly on the CLIP visual class token ($f_{cls}$), penalizing domain-discriminative visual encoding without needing a text encoder during retrieval.
- **PartVisibilityGAT:** Exploits CLIP patch tokens arranged on a $16 \times 8$ grid, partitioned into coarse vertical regions (Head, Torso, Legs), contextualized via a 3-node Graph Attention Network, and gated by a learned internal reconstruction gate.
- **Explicit Visibility Supervision:** Trains a visibility predictor supervised by exact synthetic-mask occlusion ratios ($m_{ij} \in [0, 1]$), ensuring that occluded features are down-weighted rather than misleading the metric space.
- **Strict Source-Only DG Setting:** Targets are completely unseen during training ($D_T \notin D_{train}$); zero target pseudo-labels, test-time adaptation, or fine-tuning.

---

## 2. Architecture & Pipeline

```
                                  [Input: 256x128 Pedestrian Crop]
                                                 │
                                                 ▼
                                     ┌─────────────────────────┐
                                     │  CLIP ViT-B/16 Backbone │
                                     │  (16x8 Patch Grid: 128) │
                                     └───────────┬─────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             [Class Token f_cls ∈ R^768]                     [Patch Tokens F_patch ∈ R^128x768]
                        │                                                 │
            ┌───────────┴────────────┐                                    ▼
            │                        │                        ┌───────────────────────┐
            ▼                        ▼                        │  Coarse Part Pooling  │
    ┌───────────────┐        ┌───────────────┐                │ (R1:Head, R2:Torso,   │
    │ Global Branch │        │ Domain Branch │                │  R3:Legs: 3x768)      │
    │ Linear 768->512        │ GRL α(e)      │                └───────────┬───────────┘
    │ BNNeck        │        │ Linear 768->384                            ▼
    │ L_id^g, L_tri^g        │ BN + ReLU     │                ┌───────────────────────┐
    └───────┬───────┘        │ Linear 384->M │                │ Graph Attention (GAT) │
            │                │ L_dom (CE)    │                │ Contextual Exchange   │
            │                └───────────────┘                └───────────┬───────────┘
            │                                                             ▼
            │                                                 ┌───────────────────────┐
            │                                                 │ Reconstruction Gate & │
            │                                                 │ Visibility Predictor  │
            │                                                 │ L_occ (Soft BCE)      │
            │                                                 └───────────┬───────────┘
            │                                                             ▼
            │                                                 ┌───────────────────────┐
            │                                                 │ Local Branch          │
            │                                                 │ Linear 768->512       │
            │                                                 │ BNNeck                │
            │                                                 │ L_id^l, L_tri^l       │
            │                                                 └───────────┬───────────┘
            │                                                             │
            └──────────────────────────────┬──────────────────────────────┘
                                           ▼
                             ┌───────────────────────────┐
                             │   Global-Local Fusion     │
                             │   f = (g + l) / ||g + l||_2│
                             │   (512-dim Normalized)    │
                             └───────────────────────────┘
```

---

## 3. Mathematical Formulations

### 3.1 Global Identity Objectives
- **Label-Smoothed Classification Loss ($\mathcal{L}_{id}^g$):**
  $$\mathcal{L}_{id}^g = -\frac{1}{B} \sum_{i=1}^B \sum_{k=1}^C q_{ik} \log p_{ik}, \quad q_{ik} = (1 - \epsilon)\mathbf{1}[k = y_i] + \frac{\epsilon}{C}, \quad \epsilon = 0.1$$
- **Batch-Hard Triplet Loss ($\mathcal{L}_{tri}^g$):**
  $$\mathcal{L}_{tri}^g = \frac{1}{B} \sum_{i=1}^B \left[ m + \max_{p} d(g_i, g_{i, p}^+) - \min_{n} d(g_i, g_{i, n}^-) \right]_+, \quad m = 0.3$$

### 3.2 Visual Domain Erasure Objective
- **Adversarial GRL Loss ($\mathcal{L}_{dom}$):**
  $$\mathcal{L}_{dom} = -\frac{1}{B} \sum_{i=1}^B \log P_\phi(d_i \mid R_{\alpha(e)}(f_{cls}^i))$$
  where $R_\alpha(x) = x, \; \frac{\partial R_\alpha}{\partial x} = -\alpha I$, with warm-up schedule:
  $$\alpha(e) = \min\left(1, \frac{e - 30}{10}\right) \quad \text{for } e \ge 31, \quad \alpha(e) = 0 \quad \text{for } e < 31$$

### 3.3 Visibility Supervision Objective
- **Synthetic Part Visibility Loss ($\mathcal{L}_{occ}$):**
  $$\mathcal{L}_{occ} = -\frac{1}{3B} \sum_{i=1}^B \sum_{j=1}^3 \left[ m_{ij} \log v_{ij} + (1 - m_{ij}) \log(1 - v_{ij}) \right]$$
  where $m_{ij} \in [0, 1]$ is the exact visible fraction from the synthetic occlusion mask and $v_{ij} = \sigma(w_2^\top \text{ReLU}(W_1 p'_{ij} + b_1) + b_2)$.

### 3.4 Latent Domain Factor Orthogonality
- **Orthogonality Loss ($\mathcal{L}_{ortho}$):**
  $$\mathcal{L}_{ortho} = \|\hat{G}^\top \hat{G} - I_K\|_F^2, \quad \hat{G} = \left[\frac{g_1}{\|g_1\|_2}, \dots, \frac{g_K}{\|g_K\|_2}\right] \in \mathbb{R}^{512 \times K}, \; K = 4$$

### 3.5 Total Composite Loss
$$\mathcal{L}_{total} = \lambda_{id}\mathcal{L}_{id}^g + \lambda_{tri}\mathcal{L}_{tri}^g + \lambda_{id}^{loc}\mathcal{L}_{id}^l + \lambda_{tri}^{loc}\mathcal{L}_{tri}^l + \lambda_{occ}\mathcal{L}_{occ} + \lambda_{adv}\mathcal{L}_{dom} + \lambda_{ortho}\mathcal{L}_{ortho}$$

Default Weights: $\lambda_{id}=1.0, \; \lambda_{tri}=1.0, \; \lambda_{id}^{loc}=0.5, \; \lambda_{tri}^{loc}=0.5, \; \lambda_{occ}=0.3, \; \lambda_{adv}=0.1, \; \lambda_{ortho}=0.05$.

---

## 4. 3-Stage Training Schedule

| Stage | Epochs | Trainable Modules | Purpose | GRL $\alpha$ |
| :---: | :---: | :--- | :--- | :---: |
| **Stage 1** | 1–10 | Projection heads ($W_g, W_l$), BNNecks, Classifiers, Part GAT, Visibility Head, Domain Factors | Stabilize newly introduced feature heads while CLIP visual backbone is frozen | $\alpha = 0$ |
| **Stage 2** | 11–30 | Stage 1 modules + Final 2 ViT transformer blocks (blocks 10 & 11) | Adapt high-level visual representation to Re-ID without adversarial disruption | $\alpha = 0$ |
| **Stage 3** | 31–60 | Stage 2 modules + Domain Classifier with GRL active | Joint optimization with gradual GRL warm-up | $\alpha = \min(1, \frac{e-30}{10})$ |

---

## 5. Experimental Results (Report Truth)

### Table 7: Component-Wise Ablation (Market-1501 Target)
| Configuration | Visual GRL | Part Branch | Rank-1 (%) | mAP (%) |
| :--- | :---: | :---: | :---: | :---: |
| CILP-FGDI Baseline | No | No | 80.1 | 57.1 |
| GRL-Only | Yes | No | 82.2 | 59.5 |
| Part-Only | No | Yes | 81.0 | 58.5 |
| **Full DG-ReID** | **Yes** | **Yes** | **83.5** | **61.2** |

### Table 8: Rank-1 (%) Under Increasing Target Occlusion
| Configuration | Clean (0%) | Mild (20%) | Moderate (35%) | Severe (55%) |
| :--- | :---: | :---: | :---: | :---: |
| Global Baseline | 80.1 | 72.0 | 55.0 | 32.0 |
| + Part Branch (No Visibility) | 80.5 | 74.5 | 61.0 | 41.0 |
| **Full DG-ReID** | **83.5** | **80.0** | **71.5** | **55.0** |

*Under severe occlusion, DG-ReID retains a **+23.0% Rank-1 advantage** over the global baseline.*

---

## 6. Project Structure

```
.
├── configs/                          # Experiment & ablation configurations
│   ├── default.yaml                  # Default hyperparams (P=16, K=4, B=64)
│   ├── market1501_dg.yaml            # Leave-Market-1501-out config
│   ├── msmt17_dg.yaml                # Leave-MSMT17-out config
│   └── ablations/                    # Configs for Table 7 & Table 8 ablations
│       ├── baseline_clip.yaml
│       ├── grl_only.yaml
│       ├── part_only.yaml
│       ├── full_dg_reid.yaml
│       ├── no_visibility_gat.yaml
│       ├── no_synthetic_occ.yaml
│       └── no_ortho.yaml
├── src/                              # Source code package
│   ├── models/                       # Model definitions
│   │   ├── clip_vit.py               # CLIP ViT-B/16 with bicubic pos interpolation
│   │   ├── grl.py                    # Gradient Reversal Layer with warmup
│   │   ├── domain_classifier.py      # MLP: 768 -> 384 -> M
│   │   ├── part_gat.py               # 3-part coarse pooling, GAT, vis predictor
│   │   ├── bnneck.py                 # Strong baseline BNNeck
│   │   ├── orthogonal_factors.py     # Orthogonal domain factors K=4
│   │   └── dg_reid.py                # Unified dual-branch DG-ReID model
│   ├── losses/                       # Loss functions
│   │   ├── cross_entropy.py          # Label-smoothed Cross Entropy
│   │   ├── triplet.py                # Batch-Hard Triplet loss
│   │   ├── visibility_loss.py        # Soft BCE visibility loss
│   │   ├── orthogonality_loss.py     # Frobenius norm orthogonality loss
│   │   └── total_loss.py             # Composite 7-term loss manager
│   ├── data/                         # Data loading & processing
│   │   ├── transforms.py             # Transforms (Resize, Flip, PadCrop, Normalize)
│   │   ├── occlusion.py              # Synthetic occlusion & 16x8 mask generator
│   │   ├── dataset.py                # Re-ID dataset & Multi-domain loader
│   │   ├── sampler.py                # MultiDomain PK Sampler (P=16, K=4)
│   │   ├── synthetic_data.py         # On-the-fly synthetic dataset generator
│   │   └── prepare_datasets.py       # Dataset download & verification helper
│   ├── training/                     # Training engine
│   │   ├── stage_manager.py          # 3-stage parameter freezing manager
│   │   └── trainer.py                # AMP training loop, logging & checkpointing
│   ├── evaluation/                   # Evaluation engine
│   │   ├── metrics.py                # CMC (Rank-1, 5, 10, 20) & mAP (Market protocol)
│   │   └── evaluator.py              # Query-gallery retrieval under occlusion
│   └── utils/                        # Utilities
│       ├── config.py                 # YAML config parser
│       ├── logger.py                 # Formatted console/file logger
│       ├── checkpoint.py             # Checkpoint save/load with metadata
│       ├── visualizer.py             # Markdown/LaTeX table and plot generators
│       └── reproducibility.py        # Deterministic seeding & env provenance
├── tests/                            # Comprehensive unit & integration tests
│   ├── test_backbone.py              # ViT output shapes (B, 768) and (B, 128, 768)
│   ├── test_grl.py                   # GRL identity and negative gradient verification
│   ├── test_part_gat.py              # GAT interaction and visibility prediction
│   ├── test_occlusion.py             # Synthetic mask and soft visibility ratios
│   ├── test_losses.py                # Finiteness & gradient checks for all losses
│   ├── test_stages.py                # Parameter freezing/unfreezing verification
│   ├── test_metrics.py               # CMC/mAP calculation & camera exclusion rules
│   ├── test_checkpoint.py            # Checkpoint save/load numerical tolerance
│   └── test_full_pipeline.py         # End-to-end 3-stage integration smoke test
├── scripts/                          # Bash automation scripts
│   ├── run_all_stages.sh             # Executes Stage 1, 2, 3 sequentially
│   ├── run_ablations.sh              # Runs full ablation suite
│   └── setup_environment.sh          # Virtualenv & dependency setup
├── train.py                          # Training CLI entry point
├── evaluate.py                       # Evaluation CLI entry point
├── ablate.py                         # Ablation runner CLI
├── inference.py                      # Single-image feature extraction & ranking CLI
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── LICENSE                           # MIT License
└── README.md                         # Project documentation
```

---

## 7. Installation & Verification

### 7.1 Setup Environment
```bash
git clone https://github.com/kapishverma/dg-reid.git
cd dg-reid

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 7.2 Run Automated Test Suite
Run all 22 unit and integration tests (executes in ~25 seconds on CPU):
```bash
python3 -m unittest discover tests -v
```

---

## 8. Dataset Preparation & Benchmark Directory Structures

The codebase features automatic directory resolution, supporting exact naming conventions from Kaggle and GitHub sources without manual folder renaming:

### 8.1 Benchmark Sources & Directory Layouts

#### 1. Market-1501
- **Source:** [Kaggle: Market-1501](https://www.kaggle.com/datasets/pengcw1/market-1501/data)
- **Directory:** `./data/Market-1501-v15.09.15/` (or `./data/Market-1501/`)
```
Market-1501-v15.09.15/
├── bounding_box_test/
├── bounding_box_train/
├── gt_bbox/
├── gt_query/
├── query/
└── readme.txt
```

#### 2. MSMT17
- **Source:** [Kaggle: MSMT17](https://www.kaggle.com/datasets/ouassimaazzouzi/msmt17)
- **Directory:** `./data/MSMT17_V1/` (or `./data/MSMT17/`)
```
MSMT17_V1/
├── test/
├── train/
├── list_gallery.txt
├── list_query.txt
├── list_train.txt
└── list_val.txt
```

#### 3. CUHK-SYSU
- **Source:** [Kaggle: CUHK-SYSU](https://www.kaggle.com/datasets/manaschaiaonon/cuhk-sysu)
- **Directory:** `./data/cuhk_sysu/` (or `./data/CUHK-SYSU/`)
```
cuhk_sysu/
├── Image/
│   └── SSM/
├── annotation/
│   ├── test/
│   ├── Images.mat
│   ├── Person.mat
│   └── pool.mat
└── README.txt
```

#### 4. CUHK03 (CUHK03-NP Protocol)
- **Source:** [Kaggle: CUHK03](https://www.kaggle.com/datasets/priyanagda/cuhk03) with [zhunzhong07/person-re-ranking CUHK03-NP protocol files](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP)
- **Directory:** `./data/cuhk03/` (or `./data/archive/`)
```
cuhk03/
├── cuhk03_release/
│   ├── README.md
│   └── cuhk-03.mat
├── images_detected/
├── images_labeled/
├── splits_new_detected.json
├── splits_new_labeled.json
└── pairs.csv
```

#### 5. Occluded-DukeMTMC
- **Source:** Converted from DukeMTMC-reID using [lightas/Occluded-DukeMTMC-Dataset](https://github.com/lightas/Occluded-DukeMTMC-Dataset)
- **Directory:** `./data/Occluded-DukeMTMC/` (or `./data/occluded_dukemtmc/`)
```
Occluded-DukeMTMC/
├── bounding_box_test/
├── bounding_box_train/
└── query/
```

### 8.2 Verifying Dataset Integrity
Verify that your dataset directory matches expected structure:
```bash
python3 src/data/prepare_datasets.py --verify Market-1501 --dir ./data
python3 src/data/prepare_datasets.py --verify MSMT17 --dir ./data
python3 src/data/prepare_datasets.py --verify CUHK03 --dir ./data
python3 src/data/prepare_datasets.py --verify CUHK-SYSU --dir ./data
python3 src/data/prepare_datasets.py --verify Occluded-DukeMTMC --dir ./data
```

---

## 9. Training Commands

### 9.1 Quick Smoke Test (Synthetic Multi-Domain Data)
Run an immediate multi-stage test without downloading any external datasets:
```bash
python train.py --config configs/default.yaml --synthetic --epochs 3
```

### 9.2 Full Multi-Stage Benchmark Training
Execute the complete 60-epoch schedule across leave-Market1501-out:
```bash
# Execute Stage 1 (Epochs 1-10: Frozen Backbone)
python train.py --config configs/market1501_dg.yaml --stage 1 --output_dir ./outputs/market_dg

# Execute Stage 2 (Epochs 11-30: Adapt Visual Representation)
python train.py --config configs/market1501_dg.yaml --stage 2 --resume ./outputs/market_dg/checkpoints/stage_1_final.pth --output_dir ./outputs/market_dg

# Execute Stage 3 (Epochs 31-60: Visual Domain Erasure with GRL)
python train.py --config configs/market1501_dg.yaml --stage 3 --resume ./outputs/market_dg/checkpoints/stage_2_final.pth --output_dir ./outputs/market_dg
```

Or automate all stages with one command:
```bash
bash scripts/run_all_stages.sh configs/market1501_dg.yaml ./outputs/market_dg
```

---

## 10. Evaluation & Ablation Commands

### 10.1 Zero-Shot Cross-Domain Evaluation
Evaluate on the unseen target domain under Clean, Mild, Moderate, and Severe occlusion:
```bash
python evaluate.py --config configs/market1501_dg.yaml --checkpoint ./outputs/market_dg/checkpoints/checkpoint_best.pth
```

### 10.2 Reproduce Ablation Tables
Display reported experimental matrices in Markdown and LaTeX:
```bash
python ablate.py --mode display
```

Run live ablation evaluations on target data:
```bash
python ablate.py --mode run --synthetic
```

---

## 11. Inference & Retrieval

### 11.1 Extract Feature Embedding
```bash
python inference.py --query ./path/to/pedestrian.jpg --checkpoint ./outputs/market_dg/checkpoints/checkpoint_best.pth
```

### 11.2 Match Query Against Gallery
```bash
python inference.py --query ./path/to/query.jpg --gallery ./path/to/gallery_folder/ --top_k 5 --checkpoint ./outputs/market_dg/checkpoints/checkpoint_best.pth
```

---

## 12. Publishing to GitHub

To initialize git and push the repository to GitHub:
```bash
git init -b main
git add .
git commit -m "feat: complete research-grade implementation of Visibility-Aware DG-ReID"
git remote add origin https://github.com/<your-username>/dg-reid.git
git push -u origin main
```
Or using the GitHub CLI:
```bash
gh repo create dg-reid --public --source=. --remote=origin --push
```

---

## 13. Citation

```bibtex
@article{bangera2026visibility,
  title={Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification},
  author={Bangera, Diya and Verma, Kapish and Rana, Rishit and Bhatia, Vandana},
  institution={Netaji Subhas University of Technology (NSUT), New Delhi},
  year={2026}
}
```
