"""Regenerate the tables that appear in README and the report.

Round 4 (Task 8): every table the README or the LaTeX report cites is
rebuilt here from the committed JSON results. Nothing is retrained. Running
this script prints the tables to stdout and writes them under
``reports/results/tables/`` so both surfaces can pull from the same source
and cannot drift apart.

Run:  python src/make_tables.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.core.results import format_ci


RESULTS = Path("reports/results")
TABLES = RESULTS / "tables"


def _load(name: str) -> dict:
    with open(RESULTS / f"{name}.json") as fh:
        return json.load(fh)


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def _num(x, digits=3):
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _ci(ci, digits=3):
    if ci is None:
        return "n/a"
    return format_ci(ci, digits=digits)


def phase1_table(p1: dict) -> str:
    naive = p1["naive_random_split"]
    chrono = p1["chronological_holdout"]
    exp = p1["expanding_window"]
    merged = p1["leave_one_merged_block_out"]["primary_model"]
    merged_dummy = p1["leave_one_merged_block_out"]["dummy_prior_diagnostic"]
    lobo = p1["leave_one_block_out_diagnostic"]["primary_model"]

    lines = [
        "| Protocol | balAcc | AUC |",
        "|---|---|---|",
        _row([
            "Naive random split (leaky)",
            _num(naive["balanced_accuracy"]),
            _num(naive.get("roc_auc")),
        ]),
        _row([
            f"Chronological 70/30 (gap={chrono['gap_windows']})",
            _num(chrono["balanced_accuracy"]),
            _num(chrono.get("roc_auc")),
        ]),
        _row([
            "Expanding-window (walk-forward)",
            _ci(exp["subject_bootstrap_ci"]["balanced_accuracy"]),
            "undefined (single-class folds)",
        ]),
        _row([
            "**Merged-LOBO (primary group protocol)**",
            _ci(merged["subject_bootstrap_ci"]["balanced_accuracy"]),
            _ci(merged["subject_bootstrap_ci"]["roc_auc"]),
        ]),
        _row([
            "Same splits, DummyClassifier(prior)",
            _ci(merged_dummy["subject_bootstrap_ci"]["balanced_accuracy"]),
            _ci(merged_dummy["subject_bootstrap_ci"]["roc_auc"]),
        ]),
        _row([
            "Native LOBO (diagnostic, single-class folds)",
            _ci(lobo["subject_bootstrap_ci"]["balanced_accuracy"]),
            "undefined",
        ]),
    ]
    return "\n".join(lines) + "\n"


def phase2_table(p2: dict, csp: dict | None = None) -> str:
    within = p2["within_subject_shuffled"]
    loro = p2.get("within_subject_leave_one_run_out")
    cross = p2["cross_subject"]

    within_ci = within["subject_bootstrap_ci"]
    cross_ci = cross["subject_bootstrap_ci"]

    lines = [
        "| Protocol | Model | balAcc | AUC |",
        "|---|---|---|---|",
        _row([
            "Within-subject shuffled trial CV",
            "LogReg band-power",
            _ci(within_ci["balanced_accuracy"]),
            _ci(within_ci["roc_auc"]),
        ]),
    ]

    if loro is not None:
        loro_ci = loro["subject_bootstrap_ci"]
        lines.append(_row([
            "Within-subject LORO (10 subject means)",
            "LogReg band-power",
            _ci(loro_ci["balanced_accuracy"]),
            _ci(loro_ci["roc_auc"]),
        ]))

    lines.append(_row([
        "Cross-subject LOSO",
        "LogReg band-power",
        _ci(cross_ci["balanced_accuracy"]),
        _ci(cross_ci["roc_auc"]),
    ]))

    if csp is not None:
        for proto_key, label in (
            ("within_subject_shuffled",
             "Within-subject shuffled trial CV"),
            ("within_subject_leave_one_run_out",
             "Within-subject LORO (10 subject means)"),
            ("cross_subject", "Cross-subject LOSO"),
        ):
            block = csp.get(proto_key)
            if not block:
                continue
            unit = block.get("unit", "subject")
            ci_block = block.get(unit + "_bootstrap_ci") or {}
            lines.append(_row([
                label,
                "CSP+LDA (extension)",
                _ci(ci_block.get("balanced_accuracy")),
                _ci(ci_block.get("roc_auc")),
            ]))

    return "\n".join(lines) + "\n"


def _write(name: str, content: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.md"
    with open(path, "w") as fh:
        fh.write(content)
    return path


def main():
    os.makedirs(TABLES, exist_ok=True)
    p1 = _load("phase1_results")
    p2 = _load("phase2_results")
    csp = None
    csp_path = RESULTS / "phase2_csp_lda_results.json"
    if csp_path.exists():
        with open(csp_path) as fh:
            csp = json.load(fh)

    t1 = phase1_table(p1)
    t2 = phase2_table(p2, csp)

    print("=== Phase 1 ===")
    print(t1)
    print("=== Phase 2 ===")
    print(t2)

    _write("phase1", t1)
    _write("phase2", t2)
    print(f"tables written to {TABLES}/")


if __name__ == "__main__":
    main()
