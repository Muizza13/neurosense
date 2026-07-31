"""Regenerate the three figures from the committed results JSON.

The previous version recomputed every number from scratch using the
pre-refactor modules, a different model (C = 0.5), globally fitted artifact
clipping, and F1 rather than balanced accuracy. The figures could therefore
disagree with the README while both were "produced by the code in this repo".

This version reads reports/results/phase1_results.json and phase2_results.json,
so the figures cannot drift from the reported numbers. Only Figure 3 fits a
model, because feature attribution has no number to read.

Run:  python src/make_figures.py
"""
import json
import sys

sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.io import arff
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.features import (
    STANDARD_BANDS,
    ArtifactClipper,
    band_power,
    make_continuous_windows,
)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK, HOT, COOL, GOOD, GREY = "#1b2a4a", "#c0392b", "#2e86c1", "#27ae60", "#95a5a6"
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

p1 = json.load(open("reports/results/phase1_results.json"))
p2 = json.load(open("reports/results/phase2_results.json"))


def ci_err(ci):
    """Asymmetric error bar from a bootstrap interval."""
    return np.array([[ci["point"] - ci["lo"]], [ci["hi"] - ci["point"]]])


# ======================================================================
# FIGURE 1: Phase 1, the split determines the answer
# ======================================================================
naive = p1["naive_random_split"]["balanced_accuracy"]
chrono = p1["chronological_holdout"]["balanced_accuracy"]
majority = p1["chronological_holdout"]["majority_baseline_accuracy"]
lobo_ci = p1["leave_one_block_out"]["subject_bootstrap_ci"]["balanced_accuracy"]
block_scores = [f["balanced_accuracy"] for f in p1["leave_one_block_out"]["folds"]]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

labels = ["Naive\nrandom split", "Chronological\nholdout", "Leave-one-\nblock-out"]
vals = [naive, chrono, lobo_ci["point"]]
errs = np.array([[0, 0, lobo_ci["point"] - lobo_ci["lo"]],
                 [0, 0, lobo_ci["hi"] - lobo_ci["point"]]])
bars = a1.bar(labels, vals, yerr=errs, capsize=5, color=[HOT, INK, INK])
a1.axhline(0.5, ls="--", color=HOT, lw=1.3)
a1.text(2.45, 0.515, "chance", color=HOT, fontsize=8.5, ha="right")
a1.axhline(majority, ls=":", color=GREY, lw=1.3)
a1.text(2.45, majority + 0.015, f"majority baseline ({majority:.2f})",
        color=GREY, fontsize=8, ha="right")
a1.set_ylabel("balanced accuracy")
a1.set_ylim(0, 0.9)
a1.set_title("Same features, same model.\nOnly the split changes.",
             fontsize=11, loc="left")
for b, v in zip(bars, vals):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.3f}",
            ha="center", fontweight="bold", fontsize=9)
a1.text(0, naive / 2, "LEAKED", ha="center", color="white",
        fontweight="bold", rotation=90, fontsize=9)

# Right panel: the per-block spread that a single number hides.
jitter = rng.uniform(-0.09, 0.09, len(block_scores))
a2.fill_between([-0.16, 0.16], lobo_ci["lo"], lobo_ci["hi"],
                color=HOT, alpha=0.12, zorder=1)
a2.hlines(lobo_ci["point"], -0.16, 0.16, color=HOT, lw=2.2, zorder=4)
a2.scatter(jitter, block_scores, s=45, color=INK, alpha=0.75, zorder=3)
a2.axhline(0.5, ls="--", color=HOT, lw=1.3)
a2.text(0.33, 0.515, "chance", color=HOT, fontsize=8.5, ha="right")
a2.set_xlim(-0.35, 0.35)
a2.set_xticks([])
a2.set_ylim(-0.08, 1.08)
a2.set_ylabel("balanced accuracy")
a2.set_title(f"Per-block scores (n = {len(block_scores)})\n"
             f"mean {lobo_ci['point']:.3f}, 95% CI "
             f"[{lobo_ci['lo']:.3f}, {lobo_ci['hi']:.3f}]",
             fontsize=10.5, loc="left")
a2.text(0, -0.03, "each block is single-class, so AUC is undefined",
        ha="center", fontsize=8.5, style="italic", color=GREY)

fig.suptitle("Phase 1  |  UCI EEG Eye State: apparent skill is an artifact of "
             "the split (100 windows, 1 subject)",
             fontweight="bold", x=0.02, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("reports/figures/fig1_phase1_leakage.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURE 2: Phase 2, within versus cross subject, with per-subject points
# ======================================================================
w_ci = p2["within_subject"]["subject_bootstrap_ci"]
c_ci = p2["cross_subject"]["subject_bootstrap_ci"]
w_folds = p2["within_subject"]["folds"]
c_folds = p2["cross_subject"]["folds"]

fig, ax = plt.subplots(figsize=(7.8, 4.8))
x = np.arange(2)
w = 0.35

ax.bar(x - w / 2,
       [w_ci["balanced_accuracy"]["point"], c_ci["balanced_accuracy"]["point"]],
       w,
       yerr=np.hstack([ci_err(w_ci["balanced_accuracy"]),
                       ci_err(c_ci["balanced_accuracy"])]),
       capsize=5, label="Balanced accuracy", color=COOL, zorder=2)
ax.bar(x + w / 2,
       [w_ci["roc_auc"]["point"], c_ci["roc_auc"]["point"]],
       w,
       yerr=np.hstack([ci_err(w_ci["roc_auc"]), ci_err(c_ci["roc_auc"])]),
       capsize=5, label="ROC-AUC", color=INK, zorder=2)

# Per-subject points, the spread the bars average over.
for xi, folds in zip(x, [w_folds, c_folds]):
    pts = [f["balanced_accuracy"] for f in folds]
    ax.scatter(np.full(len(pts), xi - w / 2)
               + rng.uniform(-0.07, 0.07, len(pts)),
               pts, s=22, color="white", edgecolor=INK,
               linewidth=0.8, zorder=3, alpha=0.95)

ax.axhline(0.5, ls="--", color=HOT, lw=1.3)
ax.text(1.52, 0.515, "chance", color=HOT, ha="right", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(["Within-subject\n(trial CV)",
                    "Cross-subject\n(leave-one-subject-out)"])
ax.set_ylim(0, 0.85)
ax.set_ylabel("score")
ax.legend(frameon=False, loc="upper right")
ax.set_title("Phase 2  |  Motor imagery: modest within subject, chance across "
             "subjects\nbars are subject means with 95% bootstrap CI, "
             "dots are individual subjects",
             fontweight="bold", fontsize=10.5, loc="left")

for xi, ci in zip(x, [w_ci, c_ci]):
    ax.text(xi - w / 2, 0.03, f"{ci['balanced_accuracy']['point']:.3f}",
            ha="center", fontweight="bold", fontsize=9, color="white")
    ax.text(xi + w / 2, 0.03, f"{ci['roc_auc']['point']:.3f}",
            ha="center", fontweight="bold", fontsize=9, color="white")

n_above = p2["within_subject"]["subject_mean"]["balanced_accuracy"]["n_above_chance"]
ax.text(0, 0.795, f"{n_above}/10 subjects above chance",
        ha="center", fontsize=8.5, style="italic", color=INK)

fig.tight_layout()
fig.savefig("reports/figures/fig2_phase2_generalization.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURE 3: interpretability contrast (the only panel that fits a model)
# ======================================================================
SFREQ1, WIN_SEC = 128.0, 1.0

df = pd.DataFrame(arff.loadarff("data/raw/EEG Eye State.arff")[0])
df["eyeDetection"] = df["eyeDetection"].astype(int)
channels = [c for c in df.columns if c != "eyeDetection"]

epochs, yw, starts = make_continuous_windows(
    df[channels].values, df["eyeDetection"].values, SFREQ1, WIN_SEC
)
split = int(0.70 * len(yw))

# Clipping fitted on the training segment only, as in phase1_eyestate.py.
clipper = ArtifactClipper().fit(epochs[:split])
Xtr, names1 = band_power(clipper.transform(epochs[:split]), SFREQ1,
                         STANDARD_BANDS, channels)
Xte, _ = band_power(clipper.transform(epochs[split:]), SFREQ1,
                    STANDARD_BANDS, channels)

model1 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=5000, random_state=RANDOM_STATE)),
]).fit(Xtr, yw[:split])

imp = permutation_importance(model1, Xte, yw[split:], n_repeats=30,
                             random_state=RANDOM_STATE,
                             scoring="balanced_accuracy")
order = imp.importances_mean.argsort()[::-1][:10]
POSTERIOR = {"O1", "O2", "P7", "P8", "T7", "T8"}
f1names = [names1[i] for i in order]
f1vals = [imp.importances_mean[i] for i in order]
f1cols = [COOL if n.split("_")[0] in POSTERIOR else HOT for n in f1names]

d = np.load("data/processed/physionet_features.npz")
X2, y2 = d["X"], d["y"].astype(int)
MOTOR = ["FC3", "FCZ", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
         "CP3", "CPZ", "CP4"]
names2 = [f"{ch}_{b}" for ch in MOTOR for b in ("mu", "beta")]

model2 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=5000, random_state=RANDOM_STATE)),
]).fit(X2, y2)
coef = model2.named_steps["model"].coef_[0]
o2 = np.argsort(np.abs(coef))[::-1][:10]
CLINE = {"C5", "C3", "C1", "CZ", "C2", "C4", "C6"}
f2names = [names2[i] for i in o2]
f2vals = [abs(coef[i]) for i in o2]
f2cols = [GOOD if n.split("_")[0] in CLINE else GREY for n in f2names]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
a1.barh(range(len(f1names))[::-1], f1vals, color=f1cols)
a1.set_yticks(range(len(f1names))[::-1])
a1.set_yticklabels(f1names, fontsize=9)
a1.set_xlabel("permutation importance (balanced accuracy)")
a1.set_title(f"Phase 1: attribution on a model scoring {chrono:.3f}\n"
             f"below chance, so this is not evidence about physiology",
             fontsize=10.5, loc="left", color=HOT)
a2.barh(range(len(f2names))[::-1], f2vals, color=f2cols)
a2.set_yticks(range(len(f2names))[::-1])
a2.set_yticklabels(f2names, fontsize=9)
a2.set_xlabel("|standardized coefficient|")
a2.set_title("Phase 2: top features are C3/C4 mu\n= sensorimotor lateralization",
             fontsize=10.5, loc="left", color=GOOD)
a1.legend(handles=[Patch(color=HOT, label="frontal"),
                   Patch(color=COOL, label="posterior")],
          frameon=False, fontsize=8, loc="lower right")
a2.legend(handles=[Patch(color=GOOD, label="central (motor)"),
                   Patch(color=GREY, label="other")],
          frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Explainability is only as trustworthy as the evaluation beneath "
             "it: uninterpretable where the model fails (left), corroborating "
             "known physiology where it works (right)",
             fontweight="bold", x=0.02, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("reports/figures/fig3_interpretability_contrast.png",
            bbox_inches="tight")
plt.close(fig)

print("figures written to reports/figures/")
print(f"  fig1 from phase1_results.json: naive={naive:.3f} "
      f"chrono={chrono:.3f} lobo={lobo_ci['point']:.3f} "
      f"[{lobo_ci['lo']:.3f}, {lobo_ci['hi']:.3f}]")
print(f"  fig2 from phase2_results.json: within balAcc="
      f"{w_ci['balanced_accuracy']['point']:.3f} | cross balAcc="
      f"{c_ci['balanced_accuracy']['point']:.3f}")
print(f"  fig3 recomputed: phase1 top = {f1names[0]}, "
      f"phase2 top = {f2names[0]}")
