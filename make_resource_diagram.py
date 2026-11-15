#!/usr/bin/env python3
"""
Resource-structure figure for the paper (Section 3): manufacturer node,
product node, website-image embedding, and what each of the two benchmark
tasks masks at scoring time. Pure schema diagram, no data-derived numbers
beyond the already-reported scale stats (8,900/72,789/112,598/4,484).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(10, 6.7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.7)
ax.axis("off")

NAVY = "#1F3B57"
STEEL = "#2F5597"
SLATE = "#5B6B7C"
AMBER = "#B8860B"
PAPER = "#FFFFFF"
PANEL = "#EEF2F6"


def box(x, y, w, h, title, lines, fc=PANEL, ec=STEEL, title_color=STEEL, lw=1.6, fontsize=9.3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="top",
            fontsize=11, fontweight="bold", color=title_color, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.58 - i * 0.30, line, ha="center", va="top",
                fontsize=fontsize, color="#222222", zorder=3)
    return b


# --- Manufacturer node (left) ---------------------------------------------
box(0.4, 1.9, 3.3, 3.9, "Manufacturer node  (8,900)",
    ["Name, NAICS description", "Employees, Industry", "Process_Capability", "Certification",
     "City, State", "", "→ 64-dim attribute feature", "(TF-IDF + one-hot, SVD)"],
    fc="#F4F7FB", ec=STEEL)

# website-image sub-box, nested visually below/inside the manufacturer column
box(0.4, 0.15, 3.3, 1.65, "Website image  (4,484 / 8,900)",
    ["has_image flag", "32-dim SVD  +  768-dim raw", "(Jina-CLIP-v1 embedding)",
     "— absent for 4,416 mfrs —"],
    fc="#FDF3E3", ec=AMBER, title_color=AMBER, fontsize=8.8)

# --- Product node (right) --------------------------------------------------
box(6.3, 3.35, 3.3, 2.45, "Product node  (72,789)",
    ["Free-text product string", "", "→ 64-dim TF-IDF-SVD feature", "(own vocabulary,", "not shared w/ manufacturer)"],
    fc="#F4F7FB", ec=STEEL)

# --- Material-capability label box (target of Task B) ----------------------
box(6.3, 0.9, 3.3, 1.75, "Material_Capability label",
    ["56-category frozen vocab", "withheld for VAL/TEST", "manufacturers (Task B)"],
    fc="#FBEAEA", ec="#B23A3A", title_color="#B23A3A")

# --- Arrows -----------------------------------------------------------------
arrow_main = FancyArrowPatch((3.7, 4.6), (6.3, 4.6), arrowstyle="-|>",
                              mutation_scale=18, linewidth=1.8, color=NAVY, zorder=1)
ax.add_patch(arrow_main)
ax.text(5.0, 4.80, "has_product", ha="center", fontsize=9.3, color=NAVY, fontweight="bold")
ax.text(5.0, 4.38, "112,598 edges", ha="center", fontsize=8.6, color=SLATE)
ax.text(5.0, 4.16, "(+ reverse edge for\nmessage passing)", ha="center", fontsize=7.6, color=SLATE)

arrow_label = FancyArrowPatch((2.05, 1.9), (6.3, 1.75), arrowstyle="-|>",
                               mutation_scale=16, linewidth=1.6, linestyle=(0, (4, 3)),
                               color="#B23A3A", zorder=1,
                               connectionstyle="arc3,rad=-0.18")
ax.add_patch(arrow_label)
ax.text(4.3, 2.55, "Material_Capability\n(masked, Task B)", ha="center", fontsize=7.8,
        color="#B23A3A", fontweight="bold", rotation=-9)

# --- Task annotations (plain text, no connector lines crossing other boxes) --
ax.text(5.0, 6.15, "Task A masks: held-out (manufacturer, product) edges\n"
                    "— RandomLinkSplit, transductive",
        ha="center", fontsize=8.6, color=NAVY,
        bbox=dict(boxstyle="round,pad=0.35", fc="#E8EEF6", ec=NAVY, lw=1.0))

ax.text(8.0, 3.0, "Task B masks: the label itself for\nVAL/TEST manufacturers — graph\nstructure stays fully visible",
        ha="center", fontsize=8.0, color="#B23A3A",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FBEAEA", ec="#B23A3A", lw=1.0))

# --- Legend ------------------------------------------------------------------
legend_elems = [
    Line2D([0], [0], color=NAVY, lw=1.8, marker=">", markersize=6, label="Graph edge (always visible)"),
    Line2D([0], [0], color="#B23A3A", lw=1.6, linestyle=(0, (4, 3)), marker=">", markersize=6,
           label="Label/attribute (masked at eval time per task)"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.04),
          ncol=2, frameon=False, fontsize=8.6)

ax.text(5.0, 0.02, "PMGraph resource schema — node features, the one typed graph edge, "
                    "website-image coverage, and the two tasks' masking targets",
        ha="center", fontsize=8.2, color="#666666", style="italic")

plt.tight_layout()
plt.savefig("figures/resource_schema.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig("figures/resource_schema.svg", bbox_inches="tight", facecolor="white")
print("wrote figures/resource_schema.png and .svg")
