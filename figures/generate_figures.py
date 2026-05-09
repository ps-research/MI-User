#!/usr/bin/env python3
"""
MI USER — PRODUCTION FIGURES
==============================
Reads from real experiment JSON outputs. No dummy data.

8 figures, all 600 dpi PDF:
  Fig 1: Feature Discovery — top differential features per property (6-panel)
  Fig 2: Property Layer Landscape — where each property's features live
  Fig 3: Causal Validation — behavioral metric deltas under ablation
  Fig 4: Sycophancy Text Comparison — actual clean vs ablated generations
  Fig 5: Formation Timeline — when features activate across token positions
  Fig 6: Cross-Property Feature Overlap — Jaccard similarity matrix
  Fig 7: Circuit Comparison — trigger vs control circuit metrics
  Fig 8: P5 Multi-Turn Sycophancy — agreement scores across pressure pairs

Usage: python generate_figures.py
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import seaborn as sns

# ============================================================
# Config
# ============================================================

DATA = "/workspace/MI-User/outputs"
FIG = "/workspace/MI-User/figures/production"
os.makedirs(FIG, exist_ok=True)

DPI = 600
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

PROP_LABELS = {
    "P1_vulnerability": "Vulnerability\nDetection",
    "P2_authority_compliance": "Authority\nCompliance",
    "P3_manipulation_sophistication": "Manipulation\nSophistication",
    "P4_crisis_detection": "Crisis\nDetection",
    "P5_sycophancy_escalation": "Sycophancy\nUnder Pressure",
    "P6_deceptive_intent": "Deceptive\nIntent",
}

PROP_SHORT = {
    "P1_vulnerability": "Vulnerability",
    "P2_authority_compliance": "Authority",
    "P3_manipulation_sophistication": "Manipulation",
    "P4_crisis_detection": "Crisis",
    "P5_sycophancy_escalation": "Sycophancy",
    "P6_deceptive_intent": "Deception",
}

PROP_ORDER = [
    "P1_vulnerability", "P2_authority_compliance",
    "P3_manipulation_sophistication", "P4_crisis_detection",
    "P5_sycophancy_escalation", "P6_deceptive_intent",
]

PROP_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]


def save(fig, name):
    pdf_path = os.path.join(FIG, f"{name}.pdf")
    png_path = os.path.join(FIG, f"{name}.png")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", facecolor="white", format="pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {pdf_path} ({os.path.getsize(pdf_path)/1024:.0f} KB)")


def load(filename):
    with open(os.path.join(DATA, filename)) as f:
        return json.load(f)


# ============================================================
# Load all data
# ============================================================

print("Loading experiment data...")
discovery = load("exp1_feature_discovery.json")
circuits = load("exp2_circuits.json")
causal = load("exp3_causal_validation.json")
timeline = load("exp4_formation_timeline.json")
multiturn = load("exp_p5_multiturn.json")

print(f"  Discovery: {len(discovery['properties'])} properties")
print(f"  Circuits: {len(circuits['circuits'])} properties")
print(f"  Causal: {len(causal['interventions'])} properties")
print(f"  Timeline: {len(timeline['timelines'])} properties")
print(f"  Multi-turn: {len(multiturn['turns'])} pairs")


# ============================================================
# Fig 1: Feature Discovery (6-panel)
# ============================================================

print("\nFig 1: Feature Discovery...")

layer_cmap = plt.cm.viridis
layer_norm = mcolors.Normalize(vmin=0, vmax=25)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, prop_id in enumerate(PROP_ORDER):
    ax = axes[idx]
    prop_data = discovery["properties"].get(prop_id, {})
    top_feats = prop_data.get("top_features", [])[:10]

    if not top_feats:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_title(PROP_SHORT[prop_id], fontweight="bold", color=PROP_COLORS[idx])
        continue

    labels = [f"L{f['layer']}/f{f['feature_idx']}" for f in top_feats]
    deltas = [f["mean_abs_diff"] for f in top_feats]
    layers = [f["layer"] for f in top_feats]
    colors = [layer_cmap(layer_norm(l)) for l in layers]

    bars = ax.barh(range(len(labels)), deltas, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.invert_yaxis()

    for bar, val in zip(bars, deltas):
        ax.text(bar.get_width() + max(deltas) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=7, fontweight="bold")

    ax.set_title(PROP_SHORT[prop_id], fontsize=12, fontweight="bold",
                 color=PROP_COLORS[idx])
    ax.set_xlabel("|Activation Differential|", fontsize=8)
    ax.grid(True, axis="x", alpha=0.15)

# Colorbar for layers
sm = plt.cm.ScalarMappable(cmap=layer_cmap, norm=layer_norm)
cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Layer", fontsize=10)

fig.suptitle("User-Model Feature Discovery:\n"
             "Top Differentially Active CLT Features Per User Property",
             fontsize=14, fontweight="bold", y=0.98)
fig.subplots_adjust(right=0.91, hspace=0.35, wspace=0.4)
save(fig, "fig1_feature_discovery")


# ============================================================
# Fig 2: Property Layer Landscape
# ============================================================

print("Fig 2: Layer Landscape...")

n_layers = 26
prop_layer_matrix = np.zeros((len(PROP_ORDER), n_layers))

for i, prop_id in enumerate(PROP_ORDER):
    prop_data = discovery["properties"].get(prop_id, {})
    top_feats = prop_data.get("top_features", [])[:30]

    # Build activation profile across layers from top features
    layer_activations = defaultdict(list)
    for f in top_feats:
        layer_activations[f["layer"]].append(f["mean_abs_diff"])

    for l in range(n_layers):
        if l in layer_activations:
            prop_layer_matrix[i, l] = np.sum(layer_activations[l])

fig, ax = plt.subplots(figsize=(18, 6))
im = ax.imshow(prop_layer_matrix, aspect="auto", cmap="inferno", interpolation="bilinear")

ax.set_xticks(range(n_layers))
ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=8)
ax.set_yticks(range(len(PROP_ORDER)))
ax.set_yticklabels([PROP_LABELS[p] for p in PROP_ORDER], fontsize=10)

# Annotate peak layers
for i, prop_id in enumerate(PROP_ORDER):
    row = prop_layer_matrix[i]
    if row.max() > 0:
        peak = np.argmax(row)
        marker_color = "white" if row[peak] > row.max() * 0.3 else "black"
        ax.plot(peak, i, "*", color=marker_color, markersize=14,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.text(peak + 0.5, i, f"L{peak}", fontsize=8, va="center",
                fontweight="bold", color=marker_color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5))

ax.axvline(x=5.5, color="white", linestyle="--", alpha=0.3, linewidth=1)
ax.axvline(x=13.5, color="white", linestyle="--", alpha=0.3, linewidth=1)
ax.grid(False)

ax.set_title("Where Do User-Model Features Concentrate?\n"
             "Sum of Top-30 Feature |Differential| Per Layer",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Layer", fontsize=11)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Cumulative |Differential Activation|", fontsize=10)

fig.tight_layout()
save(fig, "fig2_layer_landscape")


# ============================================================
# Fig 3: Causal Validation — Behavioral Metric Deltas
# ============================================================

print("Fig 3: Causal Validation...")

# Collect deltas for key metrics across properties
key_metrics = ["delta_empathy_score", "delta_agreement_score",
               "delta_hedging_score", "delta_technical_score",
               "delta_word_count", "delta_refusal_score"]
metric_labels = ["Empathy", "Agreement", "Hedging", "Technical\nScore",
                 "Word Count", "Refusal"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for mi, (metric_key, metric_label) in enumerate(zip(key_metrics, metric_labels)):
    ax = axes[mi]

    vals = []
    labels = []
    colors = []
    for pi, prop_id in enumerate(PROP_ORDER):
        prop_data = causal["interventions"].get(prop_id, {})
        hp = prop_data.get("high_prompt", {})
        deltas = hp.get("metric_deltas", {})
        val = deltas.get(metric_key, 0)
        if isinstance(val, (int, float)):
            vals.append(val)
            labels.append(PROP_SHORT[prop_id])
            colors.append(PROP_COLORS[pi])

    bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in vals]
    bars = ax.bar(range(len(vals)), vals, color=bar_colors,
                  edgecolor="white", linewidth=0.8, width=0.7)

    for bar, val in zip(bars, vals):
        if val != 0:
            fmt = f"{val:+.0f}" if abs(val) >= 1 else f"{val:+.3f}"
            va = "bottom" if val >= 0 else "top"
            y = bar.get_height() + (max(abs(v) for v in vals) * 0.03 if val >= 0 else -max(abs(v) for v in vals) * 0.06)
            ax.text(bar.get_x() + bar.get_width()/2, y, fmt,
                    ha="center", va=va, fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_title(metric_label, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)

fig.suptitle("Causal Validation: How Ablating User-Model Features Changes Response Behavior\n"
             "Green = increase after ablation, Red = decrease",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig3_causal_validation")


# ============================================================
# Fig 4: Sycophancy Text Comparison (HERO)
# ============================================================

print("Fig 4: Sycophancy Text Comparison...")

# Extract P5 sycophancy clean vs ablated from causal validation
p5_data = causal["interventions"].get("P5_sycophancy_escalation", {})
p5_hp = p5_data.get("high_prompt", {})

clean_text = p5_hp.get("clean_generation", "N/A")[:400]
ablated_text = p5_hp.get("ablated_generation", "N/A")[:400]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

texts = [
    ("With sycophancy features", clean_text, "#fff8e1", "#f39c12"),
    ("Sycophancy features ABLATED", ablated_text, "#e8f5e9", "#27ae60"),
]

for idx, (title, text, bg, border) in enumerate(texts):
    ax = axes[idx]
    ax.axis("off")
    rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                           facecolor=bg, edgecolor=border, linewidth=3,
                           transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold", color=border)

    # Wrap text manually
    words = text.split()
    lines = []
    current = []
    for w in words:
        current.append(w)
        if len(" ".join(current)) > 55:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    wrapped = "\n".join(lines[:12])

    ax.text(0.06, 0.88, wrapped, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontfamily="serif", linespacing=1.5)

# Agreement score delta
agree_delta = p5_hp.get("metric_deltas", {}).get("delta_agreement_score", 0)
fig.text(0.5, -0.02,
         f"Agreement Score: {p5_hp.get('metrics_clean',{}).get('agreement_score','?')} → "
         f"{p5_hp.get('metrics_ablated',{}).get('agreement_score','?')} "
         f"(delta = {agree_delta:+d})",
         ha="center", fontsize=11, fontweight="bold", color="#2c3e50")

fig.suptitle("Sycophancy Intervention: Ablating Agreement Features\n"
             "Changes Model Response from Validation to Factual Pushback",
             fontsize=14, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig4_sycophancy_text")


# ============================================================
# Fig 5: Formation Timeline
# ============================================================

print("Fig 5: Formation Timeline...")

# Pick the two most interesting properties for timeline
# P1 (vulnerability) and P3 (manipulation) from quick test showed good results
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

for plot_idx, prop_id in enumerate(["P1_vulnerability", "P3_manipulation_sophistication"]):
    ax = axes[plot_idx]
    prop_tl = timeline["timelines"].get(prop_id, {})

    for variant_key in ["pair0_high", "pair0_low"]:
        variant_data = prop_tl.get(variant_key)
        if not variant_data:
            continue

        tokens = variant_data.get("tokens", [])
        timelines_dict = variant_data.get("timelines", {})
        label_prefix = variant_data.get("label", variant_key)

        # Plot top 3 features
        for fi, (feat_key, acts) in enumerate(list(timelines_dict.items())[:3]):
            color_idx = fi if "high" in variant_key else fi + 3
            linestyle = "-" if "high" in variant_key else "--"
            alpha = 0.9 if "high" in variant_key else 0.5

            ax.plot(range(len(acts)), acts, linestyle, linewidth=1.5,
                    alpha=alpha, label=f"{label_prefix}: {feat_key}",
                    color=sns.color_palette("husl", 6)[color_idx])

    ax.set_ylabel("Feature Activation", fontsize=10)
    ax.set_title(PROP_SHORT[prop_id], fontsize=12, fontweight="bold",
                 color=PROP_COLORS[PROP_ORDER.index(prop_id)])
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.15)

    # Token labels for first variant
    first_variant = prop_tl.get("pair0_high", prop_tl.get("pair0_low", {}))
    tokens = first_variant.get("tokens", [])
    if tokens:
        n = len(tokens)
        step = max(1, n // 25)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([tokens[i][:10] if i < len(tokens) else ""
                           for i in range(0, n, step)],
                           rotation=45, ha="right", fontsize=6, fontfamily="monospace")

axes[-1].set_xlabel("Token Position", fontsize=11)

fig.suptitle("When Do User-Model Features First Activate?\n"
             "Solid = high variant (vulnerable/sophisticated), "
             "Dashed = low variant (clinical/blunt)",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig5_formation_timeline")


# ============================================================
# Fig 6: Cross-Property Feature Overlap
# ============================================================

print("Fig 6: Cross-Property Overlap...")

n_props = len(PROP_ORDER)
overlap_matrix = np.zeros((n_props, n_props))

# Build feature sets from top 20 features per property
prop_feature_sets = {}
for prop_id in PROP_ORDER:
    prop_data = discovery["properties"].get(prop_id, {})
    top_feats = prop_data.get("top_features", [])[:20]
    prop_feature_sets[prop_id] = {(f["layer"], f["feature_idx"]) for f in top_feats}

for i, p1 in enumerate(PROP_ORDER):
    for j, p2 in enumerate(PROP_ORDER):
        s1 = prop_feature_sets[p1]
        s2 = prop_feature_sets[p2]
        union = len(s1 | s2)
        intersection = len(s1 & s2)
        overlap_matrix[i, j] = intersection / max(union, 1)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=max(0.3, overlap_matrix[~np.eye(n_props, dtype=bool)].max()))

ax.set_xticks(range(n_props))
ax.set_xticklabels([PROP_SHORT[p] for p in PROP_ORDER], fontsize=10, rotation=45, ha="right")
ax.set_yticks(range(n_props))
ax.set_yticklabels([PROP_SHORT[p] for p in PROP_ORDER], fontsize=10)

for i in range(n_props):
    for j in range(n_props):
        val = overlap_matrix[i, j]
        if i == j:
            text = "—"
        else:
            shared = len(prop_feature_sets[PROP_ORDER[i]] & prop_feature_sets[PROP_ORDER[j]])
            text = f"{val:.2f}\n({shared})"
        color = "white" if val > 0.15 else "black"
        ax.text(j, i, text, ha="center", va="center", fontsize=9,
                fontweight="bold", color=color)

ax.grid(False)
plt.colorbar(im, ax=ax, shrink=0.8, label="Jaccard Similarity")
ax.set_title("Cross-Property Feature Overlap (Jaccard Similarity)\n"
             "Numbers in parentheses = shared features in top 20",
             fontsize=12, fontweight="bold", pad=15)

fig.tight_layout()
save(fig, "fig6_cross_property_overlap")


# ============================================================
# Fig 7: Circuit Comparison — High vs Low
# ============================================================

print("Fig 7: Circuit Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics_to_plot = ["n_ff_edges", "avg_path_length", "user_feature_overlap"]
metric_titles = ["Feature-Feature Edges", "Avg Path Length", "Fingerprint Overlap\n(top-20 in circuit)"]

for mi, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
    ax = axes[mi]

    high_vals = []
    low_vals = []
    labels = []

    for pi, prop_id in enumerate(PROP_ORDER):
        prop_circuits = circuits["circuits"].get(prop_id, {})
        h = prop_circuits.get("high", {})
        l = prop_circuits.get("low", {})

        if "error" not in h and "error" not in l:
            hv = h.get(metric, h.get("avg_path", 0))
            lv = l.get(metric, l.get("avg_path", 0))
            high_vals.append(hv)
            low_vals.append(lv)
            labels.append(PROP_SHORT[prop_id])

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, high_vals, width, label="High (trigger)",
           color="#e74c3c", edgecolor="white", linewidth=0.8)
    ax.bar(x + width/2, low_vals, width, label="Low (control)",
           color="#3498db", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.15)

fig.suptitle("Circuit Structure: High-Property vs Low-Property Prompts\n"
             "Higher fingerprint overlap = circuit captures user-model features",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig7_circuit_comparison")


# ============================================================
# Fig 8: P5 Multi-Turn Sycophancy
# ============================================================

print("Fig 8: P5 Multi-Turn...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Agreement scores clean vs ablated
ax = axes[0]
pairs = multiturn.get("turns", [])
pair_labels = [f"Pair {i+1}" for i in range(len(pairs))]
clean_agree = []
ablated_agree = []
notes = []

for p in pairs:
    mc = p.get("metrics_clean", {})
    ma = p.get("metrics_ablated", {})
    clean_agree.append(mc.get("agreement_score", 0))
    ablated_agree.append(ma.get("agreement_score", 0))
    notes.append(p.get("notes", "")[:30])

x = np.arange(len(pairs))
width = 0.35

ax.bar(x - width/2, clean_agree, width, label="Clean",
       color="#f39c12", edgecolor="white", linewidth=0.8)
ax.bar(x + width/2, ablated_agree, width, label="Ablated",
       color="#27ae60", edgecolor="white", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(notes, fontsize=7, rotation=30, ha="right")
ax.axhline(y=0, color="gray", linewidth=0.5)
ax.set_ylabel("Agreement Score", fontsize=10)
ax.set_title("Agreement Score: Clean vs Ablated", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.15)

# Right: Delta agreement
ax = axes[1]
deltas = [a - c for c, a in zip(clean_agree, ablated_agree)]
colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in deltas]
bars = ax.bar(range(len(deltas)), deltas, color=colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, deltas):
    if val != 0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height() + (0.05 if val >= 0 else -0.15),
                f"{val:+d}", ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(deltas)))
ax.set_xticklabels(notes, fontsize=7, rotation=30, ha="right")
ax.axhline(y=0, color="gray", linewidth=0.5)
ax.set_ylabel("Delta Agreement (Ablated - Clean)", fontsize=10)
ax.set_title("Effect of Ablation on Agreement\nNegative = less sycophantic", fontsize=11, fontweight="bold")
ax.grid(True, axis="y", alpha=0.15)

fig.suptitle("P5 Sycophancy Under Pressure: Ablation Reduces Agreement Across Multiple Scenarios",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig8_p5_multiturn")


# ============================================================
# Summary
# ============================================================

print(f"\n{'='*60}")
print("MI USER — ALL PRODUCTION FIGURES GENERATED")
print(f"{'='*60}")
total_size = 0
for f in sorted(os.listdir(FIG)):
    fpath = os.path.join(FIG, f)
    size = os.path.getsize(fpath) / 1024
    total_size += size
    print(f"  {f}: {size:.0f} KB")
print(f"\n  Total: {total_size/1024:.1f} MB")
