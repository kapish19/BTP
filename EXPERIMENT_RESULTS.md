# DG-ReID Experimental Evaluation & Ablation Results

Reproduced from BTP Mid-Semester Report: *Visibility-Aware Visual Domain Erasure for Generalizable Person Re-Identification*.

### Table 7: Component-Ablation Results (Rank-1 and mAP, %)

| Configuration | Visual GRL | Part Branch | Rank-1 (%) | mAP (%) |
| :--- | :---: | :---: | :---: | :---: |
| CILP-FGDI Baseline | No | No | **80.1** | **57.1** |
| GRL-Only | Yes | No | **82.2** | **59.5** |
| Part-Only | No | Yes | **81.0** | **58.5** |
| Full DG-ReID | Yes | Yes | **83.5** | **61.2** |


#### LaTeX for Table 7:
```latex
\begin{table}[h]
\centering
\caption{Final component-ablation results (Rank-1 and mAP, \%).}
\label{tab:ablation}
\begin{tabular}{lcccc}
\hline
Configuration & Visual GRL & Part Branch & Rank-1 (\%) & mAP (\%) \\
\hline
CILP-FGDI Baseline & No & No & 80.1 & 57.1 \\
GRL-Only & Yes & No & 82.2 & 59.5 \\
Part-Only & No & Yes & 81.0 & 58.5 \\
Full DG-ReID & Yes & Yes & 83.5 & 61.2 \\
\hline
\end{tabular}
\end{table}
```

---

### Table 8: Rank-1 (%) Results Under Increasing Target-Domain Occlusion

| Configuration | Clean (0%) | Mild (15%) | Moderate (30%) | Severe (50%) |
| :--- | :---: | :---: | :---: | :---: |
| Global Baseline | 80.1% | 72.0% | 55.0% | 32.0% |
| + Part Branch (No Visibility) | 80.5% | 74.5% | 61.0% | 41.0% |
| Full DG-ReID | 83.5% | 80.0% | 71.5% | 55.0% |


#### LaTeX for Table 8:
```latex
\begin{table}[h]
\centering
\caption{Final Rank-1 (\%) results under increasing target-domain occlusion.}
\label{tab:occlusion}
\begin{tabular}{lcccc}
\hline
Configuration & Clean & Mild & Moderate & Severe \\
\hline
Global Baseline & 80.1 & 72.0 & 55.0 & 32.0 \\
+ Part Branch (No Visibility) & 80.5 & 74.5 & 61.0 & 41.0 \\
Full DG-ReID & 83.5 & 80.0 & 71.5 & 55.0 \\
\hline
\end{tabular}
\end{table}
```
