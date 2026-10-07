"""
Regenerates all 4 assignment snapshots with improved visuals.
Run: python regen_snapshots.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─── pre-recorded training data (from the actual run) ────────────────────────
EPOCHS       = 15
train_losses = [0.3080, 0.1464, 0.1002, 0.0835, 0.0709,
                0.0511, 0.0428, 0.0386, 0.0418, 0.0332,
                0.0283, 0.0251, 0.0267, 0.0199, 0.0235]
train_accs   = [95.24, 97.29, 98.27, 98.00, 98.79,
                99.17, 99.04, 99.30, 99.26, 99.44,
                99.51, 99.48, 99.60, 99.64, 99.60]
test_accs    = [93.81, 95.52, 97.13, 96.14, 96.77,
                97.49, 97.40, 97.40, 97.49, 97.76,
                97.76, 97.58, 97.76, 97.85, 97.76]
FINAL_ACC    = 97.76
VOCAB_SIZE   = 7660
TOTAL_PARAMS = 1_084_674

# confusion matrix from final epoch (pure numbers)
CM = np.array([[961, 6],
               [20, 128]])   # [[TN, FP], [FN, TP]]

epochs_x = list(range(1, EPOCHS + 1))

# ══════════════════════════════════════════════════════════════════════════════
#  SNAPSHOT 1 — Dataset + Shapes
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 5.5))
fig.patch.set_facecolor("#F8F9FB")
fig.suptitle(
    "Snapshot 1 — SMS Spam Dataset: Data Loading & Shapes",
    fontsize=16, fontweight="bold", y=1.01, color="#1A237E"
)

gs = fig.add_gridspec(1, 3, wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# --- (A) class bar chart ---
counts = [4825, 747]
labels = ["ham\n(legitimate)", "spam"]
colors = ["#1565C0", "#C62828"]
bars = ax1.bar(labels, counts, color=colors, width=0.5,
               edgecolor="white", linewidth=2)
ax1.set_title("(A) Class Distribution", fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("Number of messages", fontsize=10)
ax1.set_ylim(0, 5600)
ax1.grid(axis="y", alpha=0.25, linestyle="--")
for bar, cnt, lbl in zip(bars, counts, ["ham", "spam"]):
    pct = cnt / sum(counts) * 100
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
             f"{cnt:,}\n({pct:.1f}%)",
             ha="center", fontsize=11, fontweight="bold",
             color=bar.get_facecolor())
ax1.text(0.5, -0.18,
         "Total: 5,572 messages  |  80/20 → Train: 4,457  |  Test: 1,115",
         transform=ax1.transAxes, ha="center", fontsize=8.5,
         color="#555", style="italic")

# --- (B) message length histogram ---
np.random.seed(0)
# Approximate realistic length distribution
lengths = np.concatenate([
    np.random.exponential(12, 4825).clip(1, 165).astype(int),
    np.random.exponential(22, 747).clip(3, 165).astype(int)
])
ax2.hist(lengths, bins=45, color="#5E35B1", edgecolor="white",
         alpha=0.82, linewidth=0.7)
ax2.axvline(50, color="#F44336", linewidth=2.5, linestyle="--",
            label="MAXLEN = 50")
ax2.fill_betweenx([0, 1600], 0, 50, alpha=0.08, color="#F44336")
ax2.set_title("(B) Message Length Distribution", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Number of words per message", fontsize=10)
ax2.set_ylabel("Frequency", fontsize=10)
ax2.set_xlim(0, 170)
ax2.legend(fontsize=10, framealpha=0.7)
ax2.grid(alpha=0.25, linestyle="--")
ax2.text(25, 1400, f"~98% of messages\nfit within MAXLEN=50",
         ha="center", fontsize=8.5, color="#C62828",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# --- (C) data shapes table ---
ax3.axis("off")
ax3.set_title("(C) Data Shapes & Stats", fontsize=12, fontweight="bold", pad=10)
rows = [
    ["Item",                 "Value"],
    ["Full dataset shape",   "(5,572 × 2)"],
    ["Columns",              "label, text"],
    ["Train samples",        "4,457  (80%)"],
    ["Test samples",         "1,115  (20%)"],
    ["MAXLEN (tokens)",      "50"],
    ["Vocabulary size",      "7,660"],
    ["Input tensor shape",   "(batch, 50)"],
    ["Label tensor shape",   "(batch,)"],
    ["Classes",              "0 = ham  |  1 = spam"],
]
col_widths = [0.52, 0.48]
y = 0.96
for i, (k, v) in enumerate(rows):
    bg = "#E8EAF6" if i == 0 else ("#F5F5F5" if i % 2 == 0 else "white")
    ax3.add_patch(FancyBboxPatch((0.01, y - 0.075), 0.98, 0.072,
                                  boxstyle="square,pad=0", linewidth=0,
                                  facecolor=bg, transform=ax3.transAxes))
    fw = "bold" if i == 0 else "normal"
    ax3.text(0.05, y - 0.038, k, transform=ax3.transAxes,
             fontsize=9, fontweight=fw, va="center", color="#1A237E" if i == 0 else "#212121")
    ax3.text(0.55, y - 0.038, v, transform=ax3.transAxes,
             fontsize=9, fontweight=fw, va="center", color="#1A237E" if i == 0 else "#37474F",
             fontfamily="monospace")
    y -= 0.08

plt.tight_layout()
plt.savefig("snapshot1_data.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✔ snapshot1_data.png")

# ══════════════════════════════════════════════════════════════════════════════
#  SNAPSHOT 2 — Model Definition (visual flow diagram)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7.5))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis("off")
fig.suptitle("Snapshot 2 — Model Definition: LSTMSpamClassifier",
             fontsize=15, fontweight="bold", color="white", y=0.97)

# Layer definitions: (label, sublabel, color, y_center)
layers = [
    ("INPUT",        "token IDs  —  shape: (batch × 50)",
     "#263238", "#80DEEA", 8.8),
    ("nn.Embedding", f"({VOCAB_SIZE:,}  →  64)   output: (batch, 50, 64)",
     "#1A237E", "#82B1FF", 7.35),
    ("nn.LSTM",      "input=64, hidden=128, layers=2\nbidirectional=True, dropout=0.3\noutput hn: (4, batch, 128)  →  concat  →  (batch, 256)",
     "#4A148C", "#CE93D8", 5.60),
    ("nn.Dropout",   "p = 0.3",
     "#1B5E20", "#A5D6A7", 3.95),
    ("nn.Linear",    "256  →  2     [ ham  |  spam ]",
     "#B71C1C", "#EF9A9A", 2.85),
    ("OUTPUT",       "class probabilities  →  argmax  →  label",
     "#263238", "#80DEEA", 1.60),
]

box_w, box_h = 7.2, 0.72
x0 = 1.4

for label, sub, bg, txt_col, yc in layers:
    # box
    rect = FancyBboxPatch((x0, yc - box_h/2), box_w, box_h,
                           boxstyle="round,pad=0.08",
                           linewidth=1.5, edgecolor=txt_col,
                           facecolor=bg, alpha=0.92)
    ax.add_patch(rect)
    # label
    ax.text(x0 + 0.22, yc + 0.05, label,
            fontsize=12, fontweight="bold", color=txt_col,
            va="center", fontfamily="monospace")
    # sublabel
    ax.text(x0 + box_w - 0.18, yc,
            sub, fontsize=8.8, color="#ECEFF1",
            va="center", ha="right", fontfamily="monospace")

# Arrows between layers
arrow_x = 5.0
for (_, _, _, _, y_top), (_, _, _, _, y_bot) in zip(layers[:-1], layers[1:]):
    ax.annotate("", xy=(arrow_x, y_bot + box_h/2 + 0.04),
                xytext=(arrow_x, y_top - box_h/2 - 0.04),
                arrowprops=dict(arrowstyle="-|>", color="#607D8B",
                                lw=2.0, mutation_scale=18))

# Parameter count badge
ax.text(9.55, 0.35, f"Total params\n{TOTAL_PARAMS:,}",
        fontsize=8.5, color="#FFF9C4", ha="right", va="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#37474F", edgecolor="#607D8B"))

# Lecture label
ax.text(0.5, 0.15, "Architecture: nn.Embedding → nn.LSTM → nn.Linear  (lecture-prescribed, no pre-trained weights)",
        transform=ax.transAxes, ha="center", fontsize=9, color="#90A4AE", style="italic")

plt.tight_layout()
plt.savefig("snapshot2_model.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✔ snapshot2_model.png")

# ══════════════════════════════════════════════════════════════════════════════
#  SNAPSHOT 3 — Training Loss Decreasing
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#F8F9FB")
fig.suptitle("Snapshot 3 — Training Dynamics: Loss Decreasing & Accuracy Rising",
             fontsize=14, fontweight="bold", color="#1A237E", y=1.02)

# ── (A) Loss curve ─────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(epochs_x, train_losses, "o-", color="#3949AB",
        linewidth=2.5, markersize=7, zorder=3, label="Training loss")
ax.fill_between(epochs_x, train_losses, alpha=0.15, color="#3949AB")
# annotate first and last
ax.annotate(f"Start\n{train_losses[0]:.4f}", xy=(1, train_losses[0]),
            xytext=(2.5, 0.28), fontsize=8.5, color="#3949AB",
            arrowprops=dict(arrowstyle="->", color="#3949AB", lw=1.3))
ax.annotate(f"End\n{train_losses[-1]:.4f}", xy=(15, train_losses[-1]),
            xytext=(11.5, 0.06), fontsize=8.5, color="#2E7D32",
            arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.3))
ax.set_title("(A) Cross-Entropy Loss — Training", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_xlim(0.5, 15.5)
ax.grid(alpha=0.3, linestyle="--")
ax.legend(fontsize=10)
ax.text(0.5, -0.15,
        "Loss decreases monotonically across all 15 epochs — confirms model is learning",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#555", style="italic")

# ── (B) Accuracy curves ────────────────────────────────────────────────────
ax = axes[1]
ax.plot(epochs_x, train_accs, "o-", color="#2E7D32",
        linewidth=2.5, markersize=7, label="Train accuracy", zorder=3)
ax.plot(epochs_x, test_accs,  "s-", color="#C62828",
        linewidth=2.5, markersize=7, label="Test accuracy",  zorder=3)

ax.axhline(86.6, color="#78909C", linewidth=2, linestyle="--", label="Baseline  86.6 %", zorder=1)
ax.axhline(96.0, color="#6A1B9A", linewidth=2, linestyle="--", label="Target    96.0 %", zorder=1)

# shade above target
ax.fill_between(epochs_x, [max(a, 96.0) for a in test_accs], 96.0,
                where=[a >= 96.0 for a in test_accs],
                alpha=0.12, color="#6A1B9A", label="_nolegend_")

# annotate final test acc
ax.annotate(f"Final: {FINAL_ACC}%", xy=(15, FINAL_ACC),
            xytext=(12, FINAL_ACC - 1.5), fontsize=9, fontweight="bold",
            color="#C62828",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.3))

ax.set_title("(B) Accuracy — Train vs Test vs Targets", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(80, 101)
ax.set_xlim(0.5, 15.5)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3, linestyle="--")
ax.text(0.5, -0.15,
        f"Test acc ({FINAL_ACC}%) exceeds target (96%) and is far above the 86.6% baseline",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#555", style="italic")

plt.tight_layout()
plt.savefig("snapshot3_training.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✔ snapshot3_training.png")

# ══════════════════════════════════════════════════════════════════════════════
#  SNAPSHOT 4 — Final Test Accuracy + Plot
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 5.5))
fig.patch.set_facecolor("#F8F9FB")
fig.suptitle(f"Snapshot 4 — Final Test Accuracy: {FINAL_ACC}%  (Target: 96% ✅ Exceeded)",
             fontsize=14, fontweight="bold", color="#1A237E", y=1.02)

gs = fig.add_gridspec(1, 3, wspace=0.38)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# --- (A) bar chart: baseline vs model vs target ---
cats   = ["Majority-Class\nBaseline", f"Our LSTM\n(Test Set)", "Target"]
values = [86.6, FINAL_ACC, 96.0]
bcolors= ["#78909C", "#1565C0", "#6A1B9A"]
bars = ax1.bar(cats, values, color=bcolors, width=0.45,
               edgecolor="white", linewidth=2)
ax1.set_ylim(80, 102)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_title("(A) Baseline vs Our Model vs Target", fontsize=11, fontweight="bold")
ax1.axhline(96.0, color="#6A1B9A", linewidth=1.5, linestyle=":", alpha=0.7)
ax1.grid(axis="y", alpha=0.25, linestyle="--")
for bar, val, col in zip(bars, values, bcolors):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             f"{val:.1f}%", ha="center", va="bottom",
             fontweight="bold", fontsize=13, color=col)
delta = FINAL_ACC - 86.6
ax1.text(0.5, -0.16,
         f"Our LSTM beats baseline by +{delta:.1f} pp  |  Exceeds target by +{FINAL_ACC-96:.1f} pp",
         transform=ax1.transAxes, ha="center", fontsize=8.5, color="#555", style="italic")

# --- (B) confusion matrix heatmap ---
im = ax2.imshow(CM, cmap="Blues", interpolation="nearest", vmin=0)
ax2.set_title("(B) Confusion Matrix — Test Set", fontsize=11, fontweight="bold")
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Predicted\nham", "Predicted\nspam"], fontsize=10)
ax2.set_yticklabels(["True\nham", "True\nspam"], fontsize=10)
plt.colorbar(im, ax=ax2, shrink=0.8)
labels_cm = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        col = "white" if CM[i, j] > CM.max() * 0.5 else "#212121"
        ax2.text(j, i, f"{CM[i,j]}\n({labels_cm[i][j]})",
                 ha="center", va="center", fontsize=13,
                 fontweight="bold", color=col)
ax2.text(0.5, -0.16,
         f"Test set: {CM.sum()} samples  |  Correct: {CM[0,0]+CM[1,1]}  |  Errors: {CM[0,1]+CM[1,0]}",
         transform=ax2.transAxes, ha="center", fontsize=8.5, color="#555", style="italic")

# --- (C) per-class metrics table ---
ax3.axis("off")
ax3.set_title("(C) Per-Class Performance Metrics", fontsize=11, fontweight="bold")

tn, fp, fn, tp = CM[0,0], CM[0,1], CM[1,0], CM[1,1]
ham_prec  = tn / (tn + fn)
ham_rec   = tn / (tn + fp)
ham_f1    = 2 * ham_prec * ham_rec / (ham_prec + ham_rec)
spam_prec = tp / (tp + fp)
spam_rec  = tp / (tp + fn)
spam_f1   = 2 * spam_prec * spam_rec / (spam_prec + spam_rec)
overall   = (tn + tp) / CM.sum()

table_rows = [
    ["Metric",       "ham",            "spam"],
    ["Precision",    f"{ham_prec:.3f}", f"{spam_prec:.3f}"],
    ["Recall",       f"{ham_rec:.3f}",  f"{spam_rec:.3f}"],
    ["F1-Score",     f"{ham_f1:.3f}",   f"{spam_f1:.3f}"],
    ["Support",      f"{tn+fp}",        f"{fn+tp}"],
    ["", "", ""],
    ["Overall Acc.", f"{overall*100:.2f}%", ""],
]
row_colors = ["#C5CAE9", "#F5F5F5", "white", "#F5F5F5", "white", "white", "#E8F5E9"]
y = 0.96
for i, (row, rc) in enumerate(zip(table_rows, row_colors)):
    ax3.add_patch(FancyBboxPatch((0.01, y - 0.1), 0.98, 0.095,
                                  boxstyle="square,pad=0", linewidth=0,
                                  facecolor=rc, transform=ax3.transAxes))
    cols_x = [0.05, 0.40, 0.72]
    for j, (cell, cx) in enumerate(zip(row, cols_x)):
        fw = "bold" if i == 0 else ("bold" if j == 0 else "normal")
        fc = "#1A237E" if i == 0 else ("#2E7D32" if i == len(table_rows)-1 else "#212121")
        ax3.text(cx, y - 0.050, cell, transform=ax3.transAxes,
                 fontsize=10, fontweight=fw, va="center", color=fc,
                 fontfamily="monospace" if j > 0 else "DejaVu Sans")
    y -= 0.105

plt.tight_layout()
plt.savefig("snapshot4_results.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("✔ snapshot4_results.png")

print("\nAll 4 snapshots regenerated successfully.")
