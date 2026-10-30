"""
Publication Figure Generator for:
"The Reasoning Paradox in Scientific Sandboxes: Why Test-Time Compute Fails in Autonomous Bioinformatics and How to Fix It"

Adheres to research paper visualization guidelines:
- Colorblind-safe palettes (Okabe-Ito / Wong 2011 Nature Methods)
- High typographical legibility with adequate vertical and horizontal whitespace
- Redundant encodings (color + shape/hatch)
- Publication-quality 300 DPI vector PDF and high-res PNG outputs
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set global matplotlib publication typography & styling
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = False
plt.rcParams['axes.edgecolor'] = '#2b2b2b'
plt.rcParams['axes.linewidth'] = 0.9
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.6

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
LATEX_FIG_DIR = REPO_ROOT / "latex" / "figures"
ARTIFACT_FIG_DIR = Path("/Users/redeemersalamiokekale/.gemini/antigravity/brain/32b9dfbf-d404-48c3-a4c2-519c881c689f/figures")

OUTPUT_DIRS = [FIGURES_DIR, LATEX_FIG_DIR]
if ARTIFACT_FIG_DIR.parent.exists():
    OUTPUT_DIRS.append(ARTIFACT_FIG_DIR)


def save_fig(fig, name):
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        pdf_path = d / f"{name}.pdf"
        png_path = d / f"{name}.png"
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved {name} (PDF & PNG) to target directories.")


# -----------------------------------------------------------------------------
# Figure 1: BioRender System Architecture
# -----------------------------------------------------------------------------
def process_figure_1():
    """
    Integrates the BioRender publication diagram into Figure 1.
    Converts and saves at 300 DPI vector PDF and PNG.
    """
    biorender_src = REPO_ROOT / "biorender.png"
    if not biorender_src.exists():
        print(f"[!] Warning: {biorender_src} not found.")
        return

    img = Image.open(biorender_src).convert("RGB")
    
    # Save to all directories
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        # Direct high-res PNG copy
        png_dest = d / "figure1_architecture.png"
        img.save(png_dest, format="PNG", dpi=(300, 300))
        
        # Vector PDF wrapper
        pdf_dest = d / "figure1_architecture.pdf"
        img.save(pdf_dest, format="PDF", resolution=300.0)

    print(f"[✓] Figure 1 BioRender successfully processed into PDF and PNG.")


# -----------------------------------------------------------------------------
# Figure 2: Benchmark Accuracy (Okabe-Ito Colorblind-Safe Palette)
# -----------------------------------------------------------------------------
def plot_figure_2():
    """
    Figure 2: Empirical accuracy on BixBench Diagnostic Core (n=22 per arm).
    Uses Okabe-Ito colorblind palette:
      - Vanilla R1: Vermilion (#D55E00)
      - Vanilla Coder: Deep Blue (#0072B2)
      - DTDR R1: Bluish Green (#009E73)
      - GPT-4o: Charcoal Slate (#4A4A4A)
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    models = [
        "Vanilla R1-14B\n(n=22)",
        "Vanilla Coder-14B\n(n=22)",
        "DeepSeek-R1 + DTDR\n(Ours, n=22)",
        "GPT-4o Baseline\n(Mitchener et al., 2025)"
    ]
    accuracies = [0.0, 4.55, 9.09, 21.0]

    # Okabe-Ito colors with distinct luminance and saturation
    colors = ['#D55E00', '#0072B2', '#009E73', '#4A4A4A']
    hatches = ['//', '', '', '..']

    bars = ax.bar(
        models,
        accuracies,
        color=colors,
        width=0.52,
        edgecolor='#1a1a1a',
        linewidth=1.2,
        zorder=3,
        hatch=hatches
    )

    # Reference threshold for GPT-4o
    ax.axhline(y=21.0, color='#4A4A4A', linestyle='--', linewidth=1.3, alpha=0.8, zorder=2)
    ax.text(
        0.05, 21.6,
        "Reported GPT-4o Multi-Turn Baseline (21.0%)",
        color='#333333',
        fontsize=9.5,
        fontweight='semibold',
        style='italic'
    )

    # Value labels above bars with clean vertical clearance
    labels = ["0.0%\n(0/22)", "4.55%\n(1/22)", "9.09%\n(2/22)", "21.0%\n(Ref)"]
    for bar, label in zip(bars, labels):
        h = bar.get_height()
        y_pos = h + 0.7 if h > 0 else 0.7
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            label,
            ha='center',
            va='bottom',
            fontsize=10.5,
            fontweight='bold',
            color='#1a1a1a',
            linespacing=1.1
        )

    # Highlight Callout for DTDR: Arrow points to left shoulder of bar to avoid intersecting text
    ax.annotate(
        "DTDR Rescues Reasoning\n(2x Coder Accuracy, 0.0% to 9.1%)",
        xy=(1.75, 7.0),
        xytext=(1.1, 15.0),
        arrowprops=dict(
            arrowstyle="->",
            lw=2.0,
            color="#009E73",
            shrinkA=0,
            shrinkB=4
        ),
        fontsize=10,
        fontweight='bold',
        color='#006d50',
        ha='center',
        bbox=dict(boxstyle="round,pad=0.45", fc="#E8F6F1", ec="#009E73", lw=1.2)
    )

    ax.set_ylabel("Diagnostic Core Accuracy (%)", fontsize=11.5, fontweight='bold', labelpad=10)
    ax.set_title("Autonomous Task Performance on BixBench Diagnostic Core (n=22, H100)", fontsize=12.5, fontweight='bold', pad=16)
    ax.set_ylim(0, 26)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_fig(fig, "figure2_accuracy_benchmark")
    plt.close()


# -----------------------------------------------------------------------------
# Figure 3: Forensic Pathology Breakdown & Step Efficiency (Colorblind-Safe)
# -----------------------------------------------------------------------------
def plot_figure_3():
    """
    Figure 3: Forensic Diagnostic Breakdown across Evaluated Trajectories.
    Panel A: Pathology rates across models (Okabe-Ito palette, clean vertical spacing).
    Panel B: Step efficiency (mean steps to termination with generous headroom).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=300, gridspec_kw={'width_ratios': [1.15, 0.85]})

    # ------------------ PANEL A: Pathologies ------------------
    pathologies = [
        "PubMed Prior Hallucination\n(Vanilla R1: Speculative Guess in Step 1)",
        "Code Execution Bug Trapping\n(Vanilla Coder: Crashes & Infinite Loops)",
        "Residual Execution Errors\n(DTDR Reasoning: 81% Error Reduction)"
    ]
    rates = [95.5, 72.7, 13.6]
    counts = ["21/22 tasks", "16/22 tasks", "3/22 tasks"]
    
    # Okabe-Ito colors: Vermilion, Deep Blue, Bluish Green
    colors_a = ['#D55E00', '#0072B2', '#009E73']
    hatches_a = ['//', '', '..']

    y_pos = np.arange(len(pathologies))
    bars1 = ax1.barh(
        y_pos,
        rates,
        color=colors_a,
        height=0.52,
        edgecolor='#1a1a1a',
        linewidth=1.1,
        zorder=3,
        hatch=hatches_a
    )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(pathologies, fontsize=9.5, fontweight='bold', linespacing=1.2)
    ax1.invert_yaxis()  # Top-down order
    ax1.set_xlabel("Observed Frequency across n=22 Tasks (%)", fontsize=11, fontweight='bold', labelpad=9)
    ax1.set_title("A. Forensic Pathology Rates across 14B Agent Sandboxes", fontsize=11.5, fontweight='bold', loc='left', pad=12)
    ax1.set_xlim(0, 108)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.6, zorder=1)
    ax1.set_axisbelow(True)

    # Clean label placement with percentage and task fraction
    for i, (v, c) in enumerate(zip(rates, counts)):
        ax1.text(
            v + 1.8,
            i,
            f"{v:.1f}%  ({c})",
            va='center',
            fontsize=10,
            fontweight='bold',
            color='#1a1a1a'
        )

    # ------------------ PANEL B: Step Efficiency ------------------
    conditions = [
        "Vanilla R1\n(Premature Exit)",
        "Vanilla Coder\n(Error Trapping)",
        "DTDR Reasoning\n(Grounded)"
    ]
    steps = [1.32, 5.18, 1.77]
    colors_b = ['#D55E00', '#0072B2', '#009E73']
    hatches_b = ['//', '', '..']

    bars2 = ax2.bar(
        conditions,
        steps,
        color=colors_b,
        width=0.48,
        edgecolor='#1a1a1a',
        linewidth=1.1,
        zorder=3,
        hatch=hatches_b
    )

    ax2.set_ylabel("Mean Steps per Trajectory", fontsize=11, fontweight='bold', labelpad=9)
    ax2.set_title("B. Trajectory Steps to Termination (n=22)", fontsize=11.5, fontweight='bold', loc='left', pad=12)
    # Generous vertical headroom to prevent crowding at the top
    ax2.set_ylim(0, 8.0)
    ax2.tick_params(axis='x', labelsize=9.5)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    ax2.set_axisbelow(True)

    # Multi-line, well-spaced annotations above each bar
    subtitles = ["Premature Guess\n(No Code)", "Trapped in Loops\n(Tracebacks)", "Fast Grounded\nConvergence"]
    for bar, val, sub in zip(bars2, steps, subtitles):
        h = bar.get_height()
        # Bold number directly above bar with 0.3 offset
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.30,
            f"{val:.2f}",
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#1a1a1a'
        )
        # Explanatory subtitle with clear vertical separation (0.95 offset above bar)
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.95,
            sub,
            ha='center',
            va='bottom',
            fontsize=8.5,
            color='#444444',
            linespacing=1.1,
            style='italic'
        )

    plt.tight_layout()
    save_fig(fig, "figure3_forensic_taxonomy")
    plt.close()


if __name__ == "__main__":
    process_figure_1()
    plot_figure_2()
    plot_figure_3()
    print("\n[✓] All publication figures regenerated with Okabe-Ito colorblind compliance and generous vertical spacing.")
