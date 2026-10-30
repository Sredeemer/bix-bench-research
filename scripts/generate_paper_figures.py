"""
Publication Figure Generator for:
"The Reasoning Paradox in Scientific Sandboxes: Why Test-Time Compute Fails in Autonomous Bioinformatics and How to Fix It"
Generates publication-quality PDF and high-res PNG figures.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Arrow

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

from pathlib import Path

# Set output directory relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
LATEX_FIG_DIR = REPO_ROOT / "latex" / "figures"

DIRS = [FIGURES_DIR, LATEX_FIG_DIR]

def save_fig(fig, name):
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        pdf_path = d / f"{name}.pdf"
        png_path = d / f"{name}.png"
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved {name} to {DIRS}")


# -----------------------------------------------------------------------------
# Figure 1: Architectural Comparison: Standard ReAct vs. Dual-Track Decoupled Reasoning (DTDR)
# -----------------------------------------------------------------------------
def plot_figure_1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1, 1]})

    # Panel A: Standard ReAct / Monolithic Failure
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title("A. Standard Single-Track Agent Loop (Failure Path)", fontsize=13, fontweight='bold', pad=15, loc='left')

    # Boxes for Panel A
    p_box = FancyBboxPatch((1, 8.2), 8, 1.2, boxstyle="round,pad=0.2", ec="#721c24", fc="#f8d7da", lw=1.5)
    ax1.add_patch(p_box)
    ax1.text(5, 8.8, "Input Prompt (Bioinformatics Task + Workspace)", ha='center', va='center', fontsize=10, fontweight='bold', color="#721c24")

    t_box = FancyBboxPatch((1, 5.4), 8, 1.8, boxstyle="round,pad=0.2", ec="#856404", fc="#fff3cd", lw=1.5)
    ax1.add_patch(t_box)
    ax1.text(5, 6.7, "Cognitive Monologue (<think> ... </think>)", ha='center', va='center', fontsize=10, fontweight='bold', color="#856404")
    ax1.text(5, 5.9, "• Falsely infers data is inaccessible ('PubMed Prior')\n• Fabricates literature-based statistical values", ha='center', va='center', fontsize=8.5, color="#533f03")

    f_box = FancyBboxPatch((1, 1.8), 8, 2.2, boxstyle="round,pad=0.2", ec="#dc3545", fc="#f5c6cb", lw=1.8)
    ax1.add_patch(f_box)
    ax1.text(5, 3.4, "Catastrophic Failure Modes", ha='center', va='center', fontsize=10.5, fontweight='bold', color="#721c24")
    ax1.text(5, 2.4, "1. Tag Collision: XML tags nested in Python (SyntaxError)\n2. Tool Flattening: Coder outputs unevaluated f-strings\n3. Memory Bloat: Unpruned monologue triggers context overflow", ha='center', va='center', fontsize=8.5, color="#721c24")

    # Arrows for Panel A
    ax1.annotate('', xy=(5, 7.3), xytext=(5, 8.2), arrowprops=dict(arrowstyle="->", lw=2, color="#721c24"))
    ax1.annotate('', xy=(5, 4.1), xytext=(5, 5.4), arrowprops=dict(arrowstyle="->", lw=2, color="#dc3545"))

    # Panel B: Dual-Track Decoupled Reasoning (DTDR - Ours)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("B. Dual-Track Decoupled Reasoning (DTDR - Ours)", fontsize=13, fontweight='bold', pad=15, loc='left')

    bp_box = FancyBboxPatch((1, 8.2), 8, 1.2, boxstyle="round,pad=0.2", ec="#155724", fc="#d4edda", lw=1.5)
    ax2.add_patch(bp_box)
    ax2.text(5, 8.8, "Input Prompt + Strict Grounding Guardrail", ha='center', va='center', fontsize=10, fontweight='bold', color="#155724")

    bt_box = FancyBboxPatch((1, 6.0), 8, 1.4, boxstyle="round,pad=0.2", ec="#0c5460", fc="#d1ecf1", lw=1.5)
    ax2.add_patch(bt_box)
    ax2.text(5, 6.9, "Dual-Track Cognitive Isolation", ha='center', va='center', fontsize=10, fontweight='bold', color="#0c5460")
    ax2.text(5, 6.3, "Step 1: Test-time reasoning formulates data inspection plan\nStep 2: <think> tokens evicted from multi-turn agent history", ha='center', va='center', fontsize=8.5, color="#0c5460")

    be_box = FancyBboxPatch((1, 3.8), 8, 1.4, boxstyle="round,pad=0.2", ec="#004085", fc="#cce5ff", lw=1.5)
    ax2.add_patch(be_box)
    ax2.text(5, 4.7, "AST/Regex Decoupled Tool Engine", ha='center', va='center', fontsize=10, fontweight='bold', color="#004085")
    ax2.text(5, 4.1, "Strips nested tags; routes code to Python bioinformatics kernel\nIntercepts submit_answer() runtime hooks directly from stdout", ha='center', va='center', fontsize=8.5, color="#004085")

    bs_box = FancyBboxPatch((1, 1.4), 8, 1.6, boxstyle="round,pad=0.2", ec="#28a745", fc="#28a745", lw=2)
    ax2.add_patch(bs_box)
    ax2.text(5, 2.3, "Verified Empirical Computation", ha='center', va='center', fontsize=11, fontweight='bold', color="white")
    ax2.text(5, 1.7, "• Solves DESeq2 / GO enrichment in 2 steps (31 sec)\n• 9.1% Verified Accuracy on n=22 Diagnostic Core (2x Coder)", ha='center', va='center', fontsize=9, color="white")

    # Arrows for Panel B
    ax2.annotate('', xy=(5, 7.5), xytext=(5, 8.2), arrowprops=dict(arrowstyle="->", lw=2, color="#155724"))
    ax2.annotate('', xy=(5, 5.3), xytext=(5, 6.0), arrowprops=dict(arrowstyle="->", lw=2, color="#0c5460"))
    ax2.annotate('', xy=(5, 3.1), xytext=(5, 3.8), arrowprops=dict(arrowstyle="->", lw=2, color="#004085"))

    plt.tight_layout()
    save_fig(fig, "figure1_architecture")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 2: Empirical Performance Across Models on BixBench Diagnostic Tasks
# -----------------------------------------------------------------------------
def plot_figure_2():
    fig, ax = plt.subplots(figsize=(8.5, 5))

    models = [
        "DeepSeek-R1-Distill-14B\n(Vanilla, n=22)",
        "Qwen 2.5 Coder 14B\n(Vanilla, n=22)",
        "DeepSeek-R1-Distill-14B\n+ DTDR (Ours, n=22)",
        "GPT-4o Baseline\n(Mitchener et al., 2025)"
    ]
    accuracies = [0.0, 4.55, 9.09, 21.0]
    colors = ['#dc3545', '#ffc107', '#28a745', '#6c757d']

    bars = ax.bar(models, accuracies, color=colors, width=0.55, edgecolor='#222222', linewidth=1.2, zorder=3)

    # Add data labels
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold', color='#111111')

    # Reference line for GPT-4o
    ax.axhline(y=21.0, color='#6c757d', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(0.1, 22.0, "GPT-4o Reference Level on BixBench Open-Answer (21.0%)", color='#555555', fontsize=9.5, fontstyle='italic')

    ax.set_ylabel("Diagnostic Task Accuracy (%)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title("Autonomous Bioinformatics Accuracy across Models (BixBench Core, H100)", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 26)
    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=1)

    # Highlight box for DTDR
    ax.annotate('DTDR Recovers Reasoning\n(2x Non-Reasoning Coder)', 
                xy=(2, 9.8), xytext=(1.2, 14.5),
                arrowprops=dict(facecolor='#28a745', shrink=0.08, width=1.5, headwidth=8),
                fontsize=10.5, fontweight='bold', color='#1e7e34')

    plt.tight_layout()
    save_fig(fig, "figure2_accuracy_benchmark")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 3: Forensic Diagnostic Taxonomy & Trajectory Pathology Breakdown
# -----------------------------------------------------------------------------
def plot_figure_3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Execution Code Errors in Multi-Turn Sandboxes
    modes = [
        "Vanilla Coder Execution Error Rate\n(16/22 tasks trapped in bugs)",
        "DTDR Reasoning Execution Error Rate\n(3/22 tasks with errors)",
        "Vanilla R1 Empirical Grounding Rate\n(1/22 tasks executed code; 95% guessed)"
    ]
    frequencies = [72.7, 13.6, 4.5]
    bar_colors = ['#d9534f', '#5cb85c', '#d9534f']

    y_pos = np.arange(len(modes))
    ax1.barh(y_pos, frequencies, color=bar_colors, edgecolor='#333333', linewidth=1.0, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(modes, fontsize=9.5, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlabel("Observed Frequency across n=22 Tasks (%)", fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title("A. Pathology Breakdown across n=22 Sweeps", fontsize=12, fontweight='bold', loc='left')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', linestyle=':', alpha=0.6, zorder=1)

    for i, v in enumerate(frequencies):
        ax1.text(v + 1.5, i, f"{v:.1f}%", va='center', fontsize=10, fontweight='bold', color='#222222')

    # Panel B: Trajectory Efficiency (Average Steps to Terminate)
    conditions = ["Vanilla R1\n(Premature Guess)", "Vanilla Coder\n(Error Trapping)", "DTDR R1\n(Decoupled Grounding)"]
    steps = [1.32, 5.18, 1.77]
    palette = ['#e74c3c', '#e67e22', '#27ae60']

    bars2 = ax2.bar(conditions, steps, color=palette, width=0.45, edgecolor='#333333', linewidth=1.0, zorder=3)
    ax2.set_ylabel("Mean Steps per Trajectory", fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title("B. Mean Execution Steps across n=22 Tasks", fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', linestyle=':', alpha=0.6, zorder=1)

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.text(0, 2.0, "Guesses in step 1", ha='center', fontsize=9, color='#c0392b', fontstyle='italic')
    ax2.text(1, 5.8, "Trapped in code bugs", ha='center', fontsize=9, color='#d35400', fontstyle='italic')
    ax2.text(2, 2.4, "Fast convergence", ha='center', fontsize=9, color='#1e8449', fontweight='bold')

    plt.tight_layout()
    save_fig(fig, "figure3_forensic_taxonomy")
    plt.close()


if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    print("All publication figures successfully created!")

