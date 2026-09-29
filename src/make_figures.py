"""Regenerate the three figures from the committed results JSON.

The previous version recomputed every number from scratch using the
pre-refactor modules, a different model (C = 0.5), globally fitted artifact
clipping, and F1 rather than balanced accuracy. The figures could therefore
disagree with the README while both were "produced by the code in this repo".

Round 4 (Task 8) went further: this script fits nothing. All three figures
read reports/results/phase1_results.json and phase2_results.json. Per-fold
signed coefficients (Figure 3) come from ``capture_coefficients=True`` on
the primary Phase 1 merged-LOBO run and the Phase 2 within-subject shuffled
CV, so the figures cannot drift from the reported numbers, and rebuilding
the figures no longer trains any model.

Run:  python src/make_figures.py
"""
import json
import sys

sys.path.insert(0, ".")

import os

import matplotlib

matplotlib.use("Agg")

# Task 7: create output directories automatically so a fresh clone does not
# need manual mkdir before running the figures.
for _d in ("reports/figures", "reports/results", "models"):
    os.makedirs(_d, exist_ok=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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
# FIGURE 1: Phase 1, the split determines the answer, with per-split baselines
# ======================================================================
naive_score = p1["naive_random_split"]["balanced_accuracy"]
naive_baselines = p1["naive_random_split"]["baselines"]

chrono = p1["chronological_holdout"]
chrono_score = chrono["balanced_accuracy"]
chrono_baselines = chrono["baselines"]
chrono_gap = chrono["gap_windows"]

expand = p1["expanding_window"]
expand_ci = expand["subject_bootstrap_ci"]["balanced_accuracy"]
expand_folds = expand["folds"]

merged = p1["leave_one_merged_block_out"]["primary_model"]
merged_ci = merged["subject_bootstrap_ci"]["balanced_accuracy"]
merged_scores = [f["balanced_accuracy"] for f in merged["folds"]]
merged_dummy = (
    p1["leave_one_merged_block_out"]["dummy_prior_diagnostic"]
    ["subject_bootstrap_ci"]["balanced_accuracy"]
)

# Retained diagnostic; not the primary group protocol.
lobo = p1["leave_one_block_out_diagnostic"]["primary_model"]
lobo_ci = lobo["subject_bootstrap_ci"]["balanced_accuracy"]
block_scores = [f["balanced_accuracy"] for f in lobo["folds"]]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

# Left: five leakage-aware protocols with per-split majority baseline where
# meaningful. Merged-LOBO is the primary group protocol; native-LOBO is
# retained only as a diagnostic (single-class folds).
protocol_labels = [
    "Naive random\nsplit (leaky)",
    f"Chronological\nholdout\n(gap = {chrono_gap})",
    "Expanding-\nwindow",
    "Leave-one-\nmerged-block-out\n(primary)",
    "Leave-one-\nnative-block-out\n(diagnostic)",
]
protocol_scores = [naive_score, chrono_score,
                   expand_ci["point"], merged_ci["point"], lobo_ci["point"]]
protocol_errs = np.array([
    [0, 0,
     expand_ci["point"] - expand_ci["lo"],
     merged_ci["point"] - merged_ci["lo"],
     lobo_ci["point"] - lobo_ci["lo"]],
    [0, 0,
     expand_ci["hi"] - expand_ci["point"],
     merged_ci["hi"] - merged_ci["point"],
     lobo_ci["hi"] - lobo_ci["point"]],
])
# The ticks are balanced accuracy of a most-frequent dummy fitted on
# training labels (naive: inside each CV fold; chronological: on the
# training prefix). Expanding-window and native LOBO stay unplotted
# because those test folds are single-class.
majority_naive = naive_baselines["most_frequent"]["balanced_accuracy"]
majority_chrono = chrono_baselines["most_frequent"]["balanced_accuracy"]
bar_colors = [HOT, INK, INK, INK, GREY]

bars = a1.bar(protocol_labels, protocol_scores, yerr=protocol_errs,
              capsize=5, color=bar_colors, zorder=2)
a1.scatter([0, 1, 3], [majority_naive, majority_chrono, merged_dummy["point"]],
           marker="_", s=1600, color=GREY, linewidths=2.5, zorder=3)
a1.text(1, majority_chrono + 0.02, f"{majority_chrono:.2f}", ha="center",
        fontsize=8.5, color=GREY)
a1.text(0, majority_naive + 0.02, f"{majority_naive:.2f}", ha="center",
        fontsize=8.5, color=GREY)
a1.text(3, merged_dummy["point"] + 0.02, f"{merged_dummy['point']:.2f}",
        ha="center", fontsize=8.5, color=GREY)
# Expanding-window and native LOBO test folds are single-class.
a1.text(2, 0.94, "single-class\nfolds", ha="center",
        fontsize=7.5, style="italic", color=GREY)
a1.text(4, 0.94, "single-class\nfolds", ha="center",
        fontsize=7.5, style="italic", color=GREY)

a1.axhline(0.5, ls="--", color=HOT, lw=1.3, zorder=1)
a1.text(len(protocol_labels) - 0.55, 0.515, "chance", color=HOT,
        fontsize=8.5, ha="right")
a1.set_ylabel("balanced accuracy")
a1.set_ylim(0, 1.05)
a1.set_title("Same features, same model.\nOnly the split, gap, and baseline change.",
             fontsize=11, loc="left")
for b, v in zip(bars, protocol_scores):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}",
            ha="center", fontweight="bold", fontsize=9)
a1.text(0, naive_score / 2, "LEAKED", ha="center", color="white",
        fontweight="bold", rotation=90, fontsize=9)

# Right panel: per-merged-block spread with the dummy-prior comparison so
# the reader can see that any pooled-metric anomaly reproduces under a
# classifier that learns nothing.
jitter = rng.uniform(-0.09, 0.09, len(merged_scores))
a2.fill_between([-0.16, 0.16], merged_ci["lo"], merged_ci["hi"],
                color=INK, alpha=0.12, zorder=1)
a2.hlines(merged_ci["point"], -0.16, 0.16, color=INK, lw=2.2, zorder=4)
a2.scatter(jitter, merged_scores, s=55, color=INK, alpha=0.85, zorder=3,
           label="primary model")
dummy_jitter = rng.uniform(-0.09, 0.09,
                           len([f["balanced_accuracy"]
                                for f in p1["leave_one_merged_block_out"]
                                ["dummy_prior_diagnostic"]["folds"]]))
dummy_scores = [f["balanced_accuracy"]
                for f in p1["leave_one_merged_block_out"]
                ["dummy_prior_diagnostic"]["folds"]]
a2.scatter(dummy_jitter + 0.02, dummy_scores, s=35, color=GREY,
           marker="s", alpha=0.9, zorder=3, label="DummyClassifier(prior)")
a2.axhline(0.5, ls="--", color=HOT, lw=1.3)
a2.text(0.33, 0.515, "chance", color=HOT, fontsize=8.5, ha="right")
a2.set_xlim(-0.35, 0.35)
a2.set_xticks([])
a2.set_ylim(-0.08, 1.08)
a2.set_ylabel("balanced accuracy")
a2.set_title(
    f"Per-fold merged-LOBO scores (n = {len(merged_scores)})\n"
    f"mean {merged_ci['point']:.3f}, 95% CI "
    f"[{merged_ci['lo']:.3f}, {merged_ci['hi']:.3f}]",
    fontsize=10.5, loc="left",
)
a2.text(0, -0.03,
        "super-block size = 4 native blocks (pre-specified)",
        ha="center", fontsize=8.5, style="italic", color=GREY)
a2.legend(frameon=False, fontsize=8, loc="upper right")

ac = p1["temporal_dependence"]["lag_report"]
ac_str = ", ".join(
    f"lag{l}={'na' if v is None else f'{v:.2f}'}"
    for l, v in zip(ac["lags"], ac["mean_abs_autocorr"])
)
gap_reason = p1["temporal_dependence"]["gap_choice"]["reason"]
n_gaps = ac.get("n_index_adjacent_pairs_with_time_gap")
gap_note = "" if n_gaps is None else f", {n_gaps} discarded-window gaps"
fig.suptitle(
    "Phase 1  |  UCI EEG Eye State, 100 windows, 1 subject\n"
    f"mean |Pearson r| {ac_str}. Gap setting {chrono_gap} windows "
    f"({gap_reason}{gap_note}). Not an independence test.",
    fontweight="bold", x=0.02, ha="left", fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("reports/figures/fig1_phase1_leakage.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURE 2: Phase 2, within versus cross subject, with per-subject points
# ======================================================================
w_ci = p2["within_subject_shuffled"]["subject_bootstrap_ci"]
c_ci = p2["cross_subject"]["subject_bootstrap_ci"]
w_folds = p2["within_subject_shuffled"]["folds"]
c_folds = p2["cross_subject"]["folds"]
loro = p2.get("within_subject_leave_one_run_out")

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
ax.set_title("Phase 2  |  within-subject trial CV and leave-one-subject-out\n"
             "bars are subject means with the documented 95% bootstrap interval, "
             "dots are individual subjects",
             fontweight="bold", fontsize=10.5, loc="left")

for xi, ci in zip(x, [w_ci, c_ci]):
    ax.text(xi - w / 2, 0.03, f"{ci['balanced_accuracy']['point']:.3f}",
            ha="center", fontweight="bold", fontsize=9, color="white")
    ax.text(xi + w / 2, 0.03, f"{ci['roc_auc']['point']:.3f}",
            ha="center", fontweight="bold", fontsize=9, color="white")

n_above = p2["within_subject_shuffled"]["subject_mean"][
    "balanced_accuracy"]["n_above_half_descriptive_count"]
ax.text(0, 0.795, f"{n_above}/10 subjects above 0.5 (descriptive count)",
        ha="center", fontsize=8.5, style="italic", color=INK)

fig.tight_layout()
fig.savefig("reports/figures/fig2_phase2_generalization.png", bbox_inches="tight")
plt.close(fig)

# ======================================================================
# FIGURE 3: signed coefficients saved with the evaluation JSON.
#
# Nothing is refit here. Phase 1 dots are the coefficients from each
# merged-block training fold. Phase 2 dots are one value per subject:
# the mean of that subject's training-fold coefficients, not a single fold.
# ======================================================================
def _load_phase1_coefs():
    merged = p1["leave_one_merged_block_out"]["primary_model"]
    names = merged["config"].get("feature_names")
    if not names:
        raise RuntimeError(
            "phase1_results.json is missing feature_names under "
            "leave_one_merged_block_out.primary_model.config. "
            "Rerun python src/phase1_eyestate.py after Task 8."
        )
    per_fold = [f["coefficients"] for f in merged["folds"]
                if "coefficients" in f]
    if not per_fold:
        raise RuntimeError(
            "phase1_results.json has no per-fold coefficients. "
            "Rerun python src/phase1_eyestate.py after Task 8."
        )
    arr = np.asarray(per_fold, dtype=float)
    return names, arr, arr.mean(axis=0)


def _load_phase2_coefs():
    within = p2["within_subject_shuffled"]
    cfg = within.get("config") or {}
    names = cfg.get("feature_names")
    if not names:
        raise RuntimeError(
            "phase2_results.json is missing feature_names under "
            "within_subject_shuffled.config. Rerun "
            "python src/phase2_motor_imagery.py after Task 8."
        )
    per_subject = [f["coefficients_mean"] for f in within["folds"]
                   if "coefficients_mean" in f]
    if not per_subject:
        raise RuntimeError(
            "phase2_results.json has no per-subject coefficients_mean. "
            "Rerun python src/phase2_motor_imagery.py after Task 8."
        )
    arr = np.asarray(per_subject, dtype=float)
    return names, arr, arr.mean(axis=0)


names1, coef_folds_p1, coef_mean_p1 = _load_phase1_coefs()
order1 = np.argsort(np.abs(coef_mean_p1))[::-1][:10]

POSTERIOR = {"O1", "O2", "P7", "P8", "T7", "T8"}
f1_labels = [names1[i] for i in order1]
f1_means = [coef_mean_p1[i] for i in order1]
f1_folds = [coef_folds_p1[:, i] for i in order1]

names2, coef_folds_p2, coef_mean_p2 = _load_phase2_coefs()
order2 = np.argsort(np.abs(coef_mean_p2))[::-1][:10]

CLINE = {"C5", "C3", "C1", "CZ", "C2", "C4", "C6"}
f2_labels = [names2[i] for i in order2]
f2_means = [coef_mean_p2[i] for i in order2]
f2_folds = [coef_folds_p2[:, i] for i in order2]


def _plot_signed_attribution(ax, labels, means, dots, colour_map,
                             xlabel, title):
    n = len(labels)
    positions = np.arange(n)[::-1]
    ax.axvline(0, color="black", lw=0.8)
    for i, (m, dot_vals) in enumerate(zip(means, dots)):
        col = colour_map(labels[i])
        ax.barh(positions[i], m, color=col, alpha=0.75, zorder=2)
        ax.scatter(dot_vals, np.full(len(dot_vals), positions[i]),
                   s=18, color="white", edgecolor="black",
                   linewidth=0.5, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left")


def _p1_colour(name):
    return COOL if name.split("_")[0] in POSTERIOR else HOT


def _p2_colour(name):
    return GOOD if name.split("_")[0] in CLINE else GREY


fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 5.4))
_plot_signed_attribution(
    a1, f1_labels, f1_means, f1_folds, _p1_colour,
    "signed standardized coefficient",
    "Phase 1. White dots: one coefficient per merged-block fold.",
)
_plot_signed_attribution(
    a2, f2_labels, f2_means, f2_folds, _p2_colour,
    "signed standardized coefficient",
    "Phase 2. White dots: subject means, not single folds.",
)
a1.legend(handles=[Patch(color=HOT, label="frontal"),
                   Patch(color=COOL, label="posterior/temporal")],
          frameon=False, fontsize=8, loc="lower right")
a2.legend(handles=[Patch(color=GOOD, label="central (motor)"),
                   Patch(color=GREY, label="other")],
          frameon=False, fontsize=8, loc="lower right")
fig.suptitle(
    "Signed standardized logistic-regression coefficients from the saved JSON. "
    "Phase 1 dots are fold coefficients. Phase 2 dots are subject means.",
    fontweight="bold", x=0.02, ha="left", fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("reports/figures/fig3_interpretability_contrast.png",
            bbox_inches="tight")
plt.close(fig)

print("figures written to reports/figures/ (no models retrained)")
print(f"  fig1 from phase1_results.json: naive={naive_score:.3f} "
      f"chrono={chrono_score:.3f} expand={expand_ci['point']:.3f} "
      f"merged-LOBO={merged_ci['point']:.3f} "
      f"[{merged_ci['lo']:.3f}, {merged_ci['hi']:.3f}]  "
      f"native-LOBO={lobo_ci['point']:.3f} (diagnostic)")
print(f"  fig2 from phase2_results.json: within balAcc="
      f"{w_ci['balanced_accuracy']['point']:.3f} | cross balAcc="
      f"{c_ci['balanced_accuracy']['point']:.3f}")
print(f"  fig3 from saved coefficients: phase1 top = {f1_labels[0]} "
      f"(mean coef {f1_means[0]:+.3f}), "
      f"phase2 top = {f2_labels[0]} (mean coef {f2_means[0]:+.3f})")
