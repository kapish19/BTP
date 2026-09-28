"""
Visualization and reporting utilities.
Generates research-grade Markdown/ASCII/LaTeX tables and degradation plots.
"""

import os
from typing import Dict, List, Optional, Union


def format_ablation_table(
    table7_data: Optional[List[Dict[str, Union[str, float]]]] = None,
    output_format: str = "markdown"
) -> str:
    """
    Formats Table 7: Final component-ablation results (Rank-1 and mAP, %).
    """
    if table7_data is None:
        # Default report values
        table7_data = [
            {"Config": "CILP-FGDI Baseline", "Visual GRL": "No", "Part Branch": "No", "Rank-1": 80.1, "mAP": 57.1},
            {"Config": "GRL-Only",           "Visual GRL": "Yes", "Part Branch": "No", "Rank-1": 82.2, "mAP": 59.5},
            {"Config": "Part-Only",          "Visual GRL": "No", "Part Branch": "Yes", "Rank-1": 81.0, "mAP": 58.5},
            {"Config": "Full DG-ReID",       "Visual GRL": "Yes", "Part Branch": "Yes", "Rank-1": 83.5, "mAP": 61.2}
        ]

    if output_format == "markdown":
        lines = [
            "| Configuration | Visual GRL | Part Branch | Rank-1 (%) | mAP (%) |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ]
        for row in table7_data:
            lines.append(f"| {row['Config']} | {row['Visual GRL']} | {row['Part Branch']} | {row['Rank-1']:.1f} | {row['mAP']:.1f} |")
        return "\n".join(lines)

    elif output_format == "latex":
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Final component-ablation results (Rank-1 and mAP, \%).}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & Visual GRL & Part Branch & Rank-1 (\%) & mAP (\%) \\",
            r"\midrule"
        ]
        for row in table7_data:
            lines.append(f"{row['Config']} & {row['Visual GRL']} & {row['Part Branch']} & {row['Rank-1']:.1f} & {row['mAP']:.1f} \\\\")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:ablation}",
            r"\end{table}"
        ])
        return "\n".join(lines)

    else:  # ASCII
        header = f"{'Configuration':<22} | {'Visual GRL':<10} | {'Part Branch':<11} | {'Rank-1 (%)':<10} | {'mAP (%)':<10}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for row in table7_data:
            lines.append(f"{row['Config']:<22} | {row['Visual GRL']:<10} | {row['Part Branch']:<11} | {row['Rank-1']:<10.1f} | {row['mAP']:<10.1f}")
        lines.append(sep)
        return "\n".join(lines)


def format_occlusion_table(
    table8_data: Optional[List[Dict[str, Union[str, float]]]] = None,
    output_format: str = "markdown"
) -> str:
    """
    Formats Table 8: Final Rank-1 (%) results under increasing target-domain occlusion.
    """
    if table8_data is None:
        table8_data = [
            {"Config": "Global Baseline",                 "Clean": 80.1, "Mild": 72.0, "Moderate": 55.0, "Severe": 32.0},
            {"Config": "+ Part Branch (No Visibility)",  "Clean": 80.5, "Mild": 74.5, "Moderate": 61.0, "Severe": 41.0},
            {"Config": "Full DG-ReID",                   "Clean": 83.5, "Mild": 80.0, "Moderate": 71.5, "Severe": 55.0}
        ]

    if output_format == "markdown":
        lines = [
            "| Configuration | Clean | Mild | Moderate | Severe |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ]
        for row in table8_data:
            lines.append(f"| {row['Config']} | {row['Clean']:.1f} | {row['Mild']:.1f} | {row['Moderate']:.1f} | {row['Severe']:.1f} |")
        return "\n".join(lines)

    elif output_format == "latex":
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Final Rank-1 (\%) results under increasing target-domain occlusion.}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & Clean & Mild & Moderate & Severe \\",
            r"\midrule"
        ]
        for row in table8_data:
            lines.append(f"{row['Config']} & {row['Clean']:.1f} & {row['Mild']:.1f} & {row['Moderate']:.1f} & {row['Severe']:.1f} \\\\")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:occlusion}",
            r"\end{table}"
        ])
        return "\n".join(lines)

    else:
        header = f"{'Configuration':<30} | {'Clean':<7} | {'Mild':<7} | {'Moderate':<8} | {'Severe':<7}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for row in table8_data:
            lines.append(f"{row['Config']:<30} | {row['Clean']:<7.1f} | {row['Mild']:<7.1f} | {row['Moderate']:<8.1f} | {row['Severe']:<7.1f}")
        lines.append(sep)
        return "\n".join(lines)


def plot_occlusion_curves(
    results: Dict[str, Dict[str, float]],
    save_path: str = "occlusion_degradation.png"
):
    """
    Plots occlusion degradation curve (Rank-1 vs occlusion severity).
    Safe import of matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib not installed; skipping plot generation.")
        return

    severities = ["clean", "mild", "moderate", "severe"]
    x = [0.0, 0.20, 0.35, 0.55]

    plt.figure(figsize=(8, 5), dpi=300)
    for model_name, data in results.items():
        y = [data.get(s, {}).get("Rank-1", 0.0) for s in severities]
        plt.plot(x, y, marker='o', linewidth=2.5, label=model_name)

    plt.title("Rank-1 Degradation Under Increasing Occlusion Area", fontsize=14)
    plt.xlabel("Target Occlusion Area Ratio", fontsize=12)
    plt.ylabel("Rank-1 Accuracy (%)", fontsize=12)
    plt.xticks(x, ["Clean (0%)", "Mild (20%)", "Moderate (35%)", "Severe (55%)"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Saved occlusion degradation plot to: {save_path}")
