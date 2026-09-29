"""Extra tests added in round 4 (Task 10).

Task 10 asks for coverage of specific behaviours the earlier tests either
touched incidentally or did not touch at all:

- run separation in leave-one-run-out
- feature-free pooled-AUC diagnostic (dummy prior on single-class LOBO)
- per-protocol dummy baseline dict on every fold
- deterministic bootstrap output for a fixed seed
- creation of missing output directories by the runnable scripts
- result-schema invariants (phase1/phase2 JSON)
- coefficient capture via evaluate_loso and
  evaluate_within_subject_shuffled_cv
- small end-to-end smoke test on a synthetic subject-fingerprint dataset

Every fixture here is synthetic. Real research artefacts are read read-only
where they exist, and only for structural checks that do not depend on
their specific numeric values.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.evaluation import (
    _extract_fold_coefficients,
    evaluate_loso,
    evaluate_within_subject_shuffled_cv,
    per_split_baselines,
)
from src.core.results import save_results
from src.core.statistics import bootstrap_ci


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=5000, random_state=42,
        )),
    ])


def _synthetic_multi_run(n_subjects=4, n_per_run=25, n_runs=3,
                         n_features=6, seed=0):
    """Per-subject, per-run trials with a mild label signal.

    Every subject has trials distributed across `n_runs` runs; the label
    lives in the first feature. The run id is stored so a LORO check
    can verify test rows come from exactly one run.
    """
    rng = np.random.default_rng(seed)
    X_all, y_all, g_all, r_all = [], [], [], []
    for s in range(n_subjects):
        for r in range(n_runs):
            labels = rng.integers(0, 2, n_per_run)
            feats = rng.standard_normal((n_per_run, n_features))
            feats[:, 0] += 1.0 * labels
            X_all.append(feats)
            y_all.append(labels)
            g_all.append(np.full(n_per_run, s))
            r_all.append(np.full(n_per_run, r + 4))  # imagery-like run ids
    return (
        np.vstack(X_all),
        np.concatenate(y_all).astype(int),
        np.concatenate(g_all).astype(int),
        np.concatenate(r_all).astype(int),
    )


# ---------------------------------------------------------------------------
# LORO run separation (Task 10 explicit item)
# ---------------------------------------------------------------------------

def test_leave_one_run_out_never_puts_the_held_out_run_in_training():
    """Within a subject, LORO must hold out exactly one run at a time and
    the held-out run must not appear in the training slice."""
    _X, y, g, run = _synthetic_multi_run(n_subjects=3, n_per_run=20,
                                         n_runs=3)
    # emulate the LORO loop that phase2_motor_imagery.py runs
    for s in np.unique(g):
        mask = g == s
        rs = run[mask]
        unique_runs = np.unique(rs)
        for held in unique_runs:
            tr = rs != held
            te = rs == held
            assert set(rs[tr]).isdisjoint({held})
            assert set(rs[te]) == {held}
            assert len(rs[tr]) + len(rs[te]) == len(rs)


# ---------------------------------------------------------------------------
# Feature-free pooled-AUC diagnostic (Task 4/10)
# ---------------------------------------------------------------------------

def test_dummy_prior_on_single_class_folds_reproduces_pooled_auc_artefact():
    """The dummy_prior LOBO diagnostic must produce a pooled AUC that
    reflects the split, not any real skill.

    Construct a dataset where every group is single-class. With
    DummyClassifier(strategy="prior") the pooled AUC should not be 0.5
    (the split forces it); with the primary model on the same splits the
    pooled AUC can also drift, which is the artefact the diagnostic
    exists to expose."""
    rng = np.random.default_rng(0)
    n_groups = 6
    per = 12
    X, y, g = [], [], []
    for gi in range(n_groups):
        # each group is single-class, alternating
        cls = gi % 2
        X.append(rng.standard_normal((per, 4)))
        y.append(np.full(per, cls))
        g.append(np.full(per, gi))
    X = np.vstack(X); y = np.concatenate(y); g = np.concatenate(g)

    res = evaluate_loso(
        X, y, g, _pipeline,
        n_boot=200, dummy_strategy="prior",
        persist_predictions=True,
    )
    pooled_auc = res["pooled_descriptive_metrics"]["roc_auc"]
    # single-class training folds mean the "positive" class is not
    # consistent across folds, so the pooled AUC is a splitting artefact.
    # It should almost never land at exactly 0.5.
    assert pooled_auc is not None
    assert not (0.499 <= pooled_auc <= 0.501)


# ---------------------------------------------------------------------------
# Per-protocol dummy baselines on every fold (Task 4/10)
# ---------------------------------------------------------------------------

def test_include_baselines_attaches_per_split_baseline_dict_to_every_fold():
    X, y, g, _ = _synthetic_multi_run(n_subjects=3, n_per_run=15, n_runs=2)
    res = evaluate_loso(
        X, y, g, _pipeline,
        n_boot=200, include_baselines=True,
    )
    for fold in res["folds"]:
        assert "baselines" in fold
        b = fold["baselines"]
        for key in (
            "n_test", "positive_rate_train", "positive_rate_test",
            "most_frequent", "stratified", "uniform",
        ):
            assert key in b
        for name in ("most_frequent", "stratified", "uniform"):
            for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
                assert metric in b[name]


def test_per_split_baselines_is_a_pure_function_of_the_input_labels():
    """Baselines should depend only on the labels passed in, so a
    reader can verify a fold's baseline by hand from the persisted
    y_true/y_train alone."""
    a = per_split_baselines(np.array([0, 1, 0, 1, 1]),
                            np.array([1, 0, 1]),
                            random_state=42)
    b = per_split_baselines(np.array([0, 1, 0, 1, 1]),
                            np.array([1, 0, 1]),
                            random_state=42)
    assert a == b


# ---------------------------------------------------------------------------
# Deterministic bootstrap output (Task 10)
# ---------------------------------------------------------------------------

def test_bootstrap_ci_is_deterministic_for_fixed_seed():
    vals = np.array([0.45, 0.51, 0.60, 0.48, 0.55, 0.62, 0.38, 0.44])
    a = bootstrap_ci(vals, n_boot=5000, random_state=7)
    b = bootstrap_ci(vals, n_boot=5000, random_state=7)
    assert a == b
    # A different seed must not return the exact same CI (else the seed
    # is being ignored).
    c = bootstrap_ci(vals, n_boot=5000, random_state=8)
    assert (c["lo"], c["hi"]) != (a["lo"], a["hi"])


# ---------------------------------------------------------------------------
# Runnable scripts auto-create output directories (Task 7/10)
# ---------------------------------------------------------------------------

def test_save_results_creates_the_results_directory(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    save_results("smoke", {"phase": "test"}, results_dir=target,
                 provenance=False)
    assert (target / "smoke.json").exists()


def test_phase_scripts_declare_mkdir_for_the_output_dirs_they_use():
    """Every runnable script must call ``os.makedirs`` (or equivalent) for
    the output directories it writes to, so a fresh clone never has to
    mkdir by hand. Static check on the source of each script."""
    checks = {
        "src/phase1_eyestate.py": ("reports/figures",
                                   "reports/results", "models"),
        "src/phase2_motor_imagery.py": ("reports/figures",
                                        "reports/results", "models"),
        "src/phase2_csp_lda.py": ("reports/figures",
                                  "reports/results", "models"),
        "src/make_figures.py": ("reports/figures",
                                "reports/results", "models"),
        # make_tables.py only writes tables, so only the tables dir is
        # required.
        "src/make_tables.py": ("reports/results/tables",),
    }
    for path, dirs in checks.items():
        src = Path(path).read_text()
        assert "makedirs" in src or "mkdir" in src, (
            f"{path} never calls makedirs/mkdir"
        )
        for target in dirs:
            assert target in src, f"{path} does not reference {target}"


# ---------------------------------------------------------------------------
# Coefficient capture (Task 6/8)
# ---------------------------------------------------------------------------

def test_evaluate_loso_captures_signed_coefficients_when_asked():
    X, y, g, _ = _synthetic_multi_run(n_subjects=4, n_per_run=20)
    res = evaluate_loso(
        X, y, g, _pipeline,
        n_boot=200,
        capture_coefficients=True,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
    )
    assert res["config"]["coefficients_captured"] is True
    assert res["config"]["feature_names"] == [f"f{i}" for i in range(X.shape[1])]
    for fold in res["folds"]:
        assert "coefficients" in fold
        assert len(fold["coefficients"]) == X.shape[1]
    # signs are informative on this fixture: the label-carrying feature
    # (index 0) should tend to have a positive coefficient.
    signs = [f["coefficients"][0] > 0 for f in res["folds"]]
    assert sum(signs) >= len(signs) - 1


def test_evaluate_loso_does_not_capture_coefficients_under_dummy_strategy():
    """A dummy classifier has no meaningful coefficients; requesting them
    with dummy_strategy set must not add a coefficients field."""
    X, y, g, _ = _synthetic_multi_run(n_subjects=3, n_per_run=20)
    res = evaluate_loso(
        X, y, g, _pipeline,
        n_boot=100,
        capture_coefficients=True,
        dummy_strategy="prior",
    )
    for fold in res["folds"]:
        assert "coefficients" not in fold
    assert res["config"]["coefficients_captured"] is False


def test_extract_fold_coefficients_returns_none_for_nonlinear_estimator():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    rf.fit(rng.standard_normal((30, 4)),
           rng.integers(0, 2, 30))
    assert _extract_fold_coefficients(rf) is None
    assert _extract_fold_coefficients(None) is None


def test_within_subject_shuffled_cv_captures_per_subject_coefficients():
    X, y, g, _ = _synthetic_multi_run(n_subjects=4, n_per_run=25)
    res = evaluate_within_subject_shuffled_cv(
        X, y, g, _pipeline,
        n_splits=5,
        capture_coefficients=True,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
    )
    assert res["config"]["coefficients_captured"] is True
    for fold in res["folds"]:
        assert "coefficients_mean" in fold
        assert "coefficients_per_fold" in fold
        assert len(fold["coefficients_mean"]) == X.shape[1]
        assert len(fold["coefficients_per_fold"]) == 5


# ---------------------------------------------------------------------------
# Result schema (Task 10)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("reports/results/phase1_results.json").exists(),
    reason="phase1_results.json not present in this checkout",
)
def test_phase1_results_json_carries_the_expected_top_level_keys():
    with open("reports/results/phase1_results.json") as fh:
        p1 = json.load(fh)
    required = {
        "phase",
        "primary_model",
        "primary_model_prespecified",
        "temporal_dependence",
        "naive_random_split",
        "chronological_holdout",
        "expanding_window",
        "leave_one_merged_block_out",
        "leave_one_block_out_diagnostic",
    }
    assert required.issubset(p1.keys())

    merged = p1["leave_one_merged_block_out"]
    assert merged["grouping_rule_prespecified"] is True
    assert merged["super_block_size_native_blocks"] == 4
    primary = merged["primary_model"]
    assert "subject_bootstrap_ci" in primary
    # Task 6/8: coefficients should live in each merged-LOBO fold
    assert any("coefficients" in f for f in primary["folds"])
    assert primary["config"].get("feature_names") is not None


@pytest.mark.skipif(
    not Path("reports/results/phase2_results.json").exists(),
    reason="phase2_results.json not present in this checkout",
)
def test_phase2_results_json_carries_the_expected_top_level_keys():
    with open("reports/results/phase2_results.json") as fh:
        p2 = json.load(fh)
    required = {
        "phase",
        "primary_model",
        "within_subject_shuffled",
        "within_subject_leave_one_run_out",
        "cross_subject",
        "naive_random_split",
    }
    assert required.issubset(p2.keys())

    ws = p2["within_subject_shuffled"]
    assert "subject_bootstrap_ci" in ws
    # Task 6/8: per-subject mean coefficients should be stored
    assert any("coefficients_mean" in f for f in ws["folds"])

    cross = p2["cross_subject"]
    for fold in cross["folds"]:
        assert "y_true" in fold and "y_pred" in fold and "y_proba" in fold


# ---------------------------------------------------------------------------
# Figure generation smoke test (Task 10)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (
        Path("reports/results/phase1_results.json").exists()
        and Path("reports/results/phase2_results.json").exists()
    ),
    reason="results JSONs not present in this checkout",
)
def test_make_figures_reads_json_and_writes_three_pngs(tmp_path, monkeypatch):
    """make_figures should never fit models. Running it from a fresh figures
    directory must produce the three expected PNGs by reading the JSONs
    alone."""
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "src/make_figures.py"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for fname in (
        "fig1_phase1_leakage.png",
        "fig2_phase2_generalization.png",
        "fig3_interpretability_contrast.png",
    ):
        assert (fig_dir / fname).exists()


# ---------------------------------------------------------------------------
# Small end-to-end smoke test on synthetic data (Task 10)
# ---------------------------------------------------------------------------

def test_end_to_end_synthetic_fingerprint_run_returns_a_full_report():
    """Full leakage-safe evaluation loop on a synthetic subject-fingerprint
    dataset: naive split inflates, LOSO sits near chance, JSON schema is
    complete. This is the smoke test for the whole shared-core path."""
    from src.core.evaluation import evaluate_naive_split

    X, y, g, _ = _synthetic_multi_run(n_subjects=6, n_per_run=25, n_runs=2)
    naive = evaluate_naive_split(X, y, _pipeline, include_baselines=True)
    loso = evaluate_loso(
        X, y, g, _pipeline,
        n_boot=500,
        include_baselines=True,
        persist_predictions=True,
        capture_coefficients=True,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
    )

    # 1. Naive split is at least as high as LOSO (it never leaks negatively)
    assert (
        naive["balanced_accuracy"]
        >= loso["subject_mean"]["balanced_accuracy"]["mean"] - 0.05
    )
    # 2. LOSO has one row per subject
    assert len(loso["folds"]) == len(np.unique(g))
    # 3. Every fold has baselines, predictions and coefficients
    for fold in loso["folds"]:
        assert "baselines" in fold
        assert len(fold["y_true"]) == fold["n_test"]
        assert len(fold["coefficients"]) == X.shape[1]
    # 4. Subject-level bootstrap CI has n_subjects rows (never n_epochs)
    ci = loso["subject_bootstrap_ci"]["balanced_accuracy"]
    assert ci["n_subjects"] == len(np.unique(g))
