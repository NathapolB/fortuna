"""EDA script — produces data/reports/phase1-eda.md and figures/.

Run after backfill.py has populated data/raw/draws.jsonl.

Usage:
    python scripts/eda.py

SPEC Phase 1 deliverable §12:
    - Total draws count, date range
    - Digit position frequency histograms (Plotly → PNG)
    - Autocorrelation of digit sums (lag 1–30)
    - Chi-square uniformity test on each digit position
    - Two-digit suffix histogram
    - Runs test
    - Honest reporting: "if signal exists it should appear here"
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
from scipy import stats

from fortuna.config import DRAWS_JSONL, REPORTS_DIR, check_not_icloud
from fortuna.store import DrawStore

FIGURES_DIR = REPORTS_DIR / "figures"
EDA_REPORT = REPORTS_DIR / "phase1-eda.md"


def run_eda() -> None:
    check_not_icloud()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    store = DrawStore(DRAWS_JSONL)
    draws = store.all_draws()

    if not draws:
        print("No draws found in draws.jsonl. Run backfill.py first.")
        sys.exit(1)

    draws.sort(key=lambda d: d.draw_id)

    print(f"Loaded {len(draws)} draws")
    print(f"Date range: {draws[0].draw_date} to {draws[-1].draw_date}")

    first_prizes = [d.first_prize for d in draws]
    two_digit_backs = [d.two_digit_back for d in draws]

    # -----------------------------------------------------------------------
    # 1. Digit position frequency analysis (positions 0–5 of first_prize)
    # -----------------------------------------------------------------------

    digit_freqs: list[Counter] = [Counter() for _ in range(6)]
    for fp in first_prizes:
        for pos, ch in enumerate(fp):
            digit_freqs[pos][ch] += 1

    # Chi-square uniformity test per position
    chi_results: list[dict] = []
    for pos in range(6):
        observed = [digit_freqs[pos].get(str(d), 0) for d in range(10)]
        n = sum(observed)
        expected = [n / 10.0] * 10
        chi2, p_val = stats.chisquare(observed, f_exp=expected)
        chi_results.append({
            "position": pos,
            "n": n,
            "chi2": chi2,
            "p_value": p_val,
            "reject_h0_at_005": p_val < 0.05,
        })

    # Plot digit position frequencies
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("First Prize Digit Frequency by Position", fontsize=14)
    for pos in range(6):
        ax = axes[pos // 3][pos % 3]
        digits = [str(d) for d in range(10)]
        counts = [digit_freqs[pos].get(d, 0) for d in digits]
        total = sum(counts)
        expected_count = total / 10
        ax.bar(digits, counts, color="steelblue", alpha=0.7, label="Observed")
        ax.axhline(expected_count, color="red", linestyle="--", label="Expected (uniform)")
        ax.set_title(f"Position {pos + 1}  (χ²={chi_results[pos]['chi2']:.2f}, p={chi_results[pos]['p_value']:.3f})")
        ax.set_xlabel("Digit")
        ax.set_ylabel("Count")
        if pos == 0:
            ax.legend()

    plt.tight_layout()
    fig_path = FIGURES_DIR / "digit_position_frequency.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # -----------------------------------------------------------------------
    # 2. Two-digit suffix histogram
    # -----------------------------------------------------------------------

    two_digit_counter = Counter(two_digit_backs)
    fig2, ax2 = plt.subplots(figsize=(16, 5))
    labels = [f"{i:02d}" for i in range(100)]
    values = [two_digit_counter.get(f"{i:02d}", 0) for i in range(100)]
    expected_two = len(draws) / 100
    ax2.bar(range(100), values, color="coral", alpha=0.7)
    ax2.axhline(expected_two, color="navy", linestyle="--", label=f"Expected ({expected_two:.1f})")
    ax2.set_title("Two-Digit Suffix (เลขท้าย 2 ตัว) Frequency Distribution")
    ax2.set_xlabel("Two-digit suffix (00–99)")
    ax2.set_ylabel("Count")
    ax2.set_xticks(range(0, 100, 10))
    ax2.set_xticklabels([f"{i:02d}" for i in range(0, 100, 10)])
    ax2.legend()
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "two_digit_suffix_histogram.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path}")

    # Chi-square for two-digit suffix
    two_obs = [two_digit_counter.get(f"{i:02d}", 0) for i in range(100)]
    two_n = sum(two_obs)
    two_expected = [two_n / 100.0] * 100
    two_chi2, two_p = stats.chisquare(two_obs, f_exp=two_expected)

    # -----------------------------------------------------------------------
    # 3. Digit sum autocorrelation
    # -----------------------------------------------------------------------

    digit_sums = [sum(int(c) for c in fp) for fp in first_prizes]
    digit_sums_arr = np.array(digit_sums, dtype=float)
    digit_sums_centered = digit_sums_arr - digit_sums_arr.mean()
    n = len(digit_sums_centered)
    max_lag = min(30, n // 2)

    acf_values = []
    for lag in range(1, max_lag + 1):
        if n - lag > 0:
            acf = np.correlate(digit_sums_centered[:n - lag], digit_sums_centered[lag:])[0]
            acf /= (np.sum(digit_sums_centered ** 2))
            acf_values.append(acf)
        else:
            acf_values.append(0.0)

    confidence_bound = 1.96 / np.sqrt(n)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    lags = list(range(1, len(acf_values) + 1))
    ax3.bar(lags, acf_values, color="teal", alpha=0.7)
    ax3.axhline(confidence_bound, color="red", linestyle="--", label=f"95% CI ±{confidence_bound:.3f}")
    ax3.axhline(-confidence_bound, color="red", linestyle="--")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_title("Autocorrelation of First Prize Digit Sum (Lag 1–30)")
    ax3.set_xlabel("Lag")
    ax3.set_ylabel("Autocorrelation")
    ax3.legend()
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "digit_sum_autocorrelation.png"
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path}")

    # -----------------------------------------------------------------------
    # 4. Runs test on digit sums (above/below median)
    # -----------------------------------------------------------------------

    median_sum = np.median(digit_sums_arr)
    signs = [1 if s >= median_sum else 0 for s in digit_sums]
    n1 = sum(signs)
    n2 = len(signs) - n1
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            runs += 1

    # Expected runs under H0 randomness
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z_runs = (runs - expected_runs) / (var_runs ** 0.5) if var_runs > 0 else 0
    p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

    # -----------------------------------------------------------------------
    # 5. Build EDA report markdown
    # -----------------------------------------------------------------------

    date_range_years = (
        (len(draws) * 14) / 365.25
    )  # approximate — 2 draws/month

    report_lines = [
        "# Phase 1 EDA Report — Project Fortuna",
        "",
        f"> Generated from {len(draws)} draws",
        f"> Date range: {draws[0].draw_date} to {draws[-1].draw_date}",
        f"> Approximate span: {date_range_years:.1f} years (based on draw count)",
        "",
        "---",
        "",
        "## Honest framing",
        "",
        "**If signal exists it should appear here.** This analysis tests whether",
        "Thai Government Lottery draws deviate from uniform random. The null",
        "hypothesis is uniformity. Most results are expected to show p > 0.05",
        "(fail to reject H0), confirming the lottery operates as designed.",
        "Any p < 0.05 finding here is a statistical anomaly that warrants",
        "Phase 2 investigation — not a claim of actionable edge.",
        "",
        "---",
        "",
        "## 1. Data Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total draws | {len(draws)} |",
        f"| First draw | {draws[0].draw_date} |",
        f"| Last draw | {draws[-1].draw_date} |",
        f"| Missing draw_id duplicates | 0 (dedup enforced by store) |",
        "",
        "---",
        "",
        "## 2. First Prize — Digit Position Frequency",
        "",
        "Chi-square uniformity test per digit position (H0: each digit 0–9 equally likely):",
        "",
        "| Position | n | χ² | p-value | Reject H0 (p<0.05)? |",
        "|----------|---|----|---------|---------------------|",
    ]

    for r in chi_results:
        reject_str = "YES — anomaly" if r["reject_h0_at_005"] else "No"
        report_lines.append(
            f"| {r['position'] + 1} | {r['n']} | {r['chi2']:.2f} | {r['p_value']:.4f} | {reject_str} |"
        )

    report_lines += [
        "",
        "![Digit position frequency](figures/digit_position_frequency.png)",
        "",
        "**Interpretation:** p > 0.05 on most/all positions = lottery looks uniform",
        "(expected). Any position with p < 0.05 is an anomaly worth investigating",
        "in Phase 2 — but requires BH-FDR correction across all tests before claiming",
        "significance (SPEC §7.2).",
        "",
        "---",
        "",
        "## 3. Two-Digit Suffix Distribution",
        "",
        f"Chi-square uniformity test on เลขท้าย 2 ตัว (00–99):",
        f"  χ² = {two_chi2:.2f}, p = {two_p:.4f}, n = {two_n}",
        f"  Reject H0: {'YES — anomaly' if two_p < 0.05 else 'No'}",
        "",
        "![Two-digit suffix histogram](figures/two_digit_suffix_histogram.png)",
        "",
        "---",
        "",
        "## 4. Autocorrelation of Digit Sums",
        "",
        f"95% confidence bounds: ±{confidence_bound:.4f}",
        "",
        "Lags outside the confidence band suggest non-random serial dependence.",
        "",
        "| Lag | ACF | Outside 95% CI? |",
        "|-----|-----|-----------------|",
    ]

    for lag, acf in zip(lags, acf_values):
        outside = "YES" if abs(acf) > confidence_bound else "No"
        report_lines.append(f"| {lag} | {acf:.4f} | {outside} |")

    report_lines += [
        "",
        "![Autocorrelation](figures/digit_sum_autocorrelation.png)",
        "",
        "---",
        "",
        "## 5. Runs Test",
        "",
        "Test whether the sequence of above/below-median digit sums is random.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| n (draws) | {len(signs)} |",
        f"| n above median | {n1} |",
        f"| n below median | {n2} |",
        f"| Observed runs | {runs} |",
        f"| Expected runs | {expected_runs:.2f} |",
        f"| Z-statistic | {z_runs:.4f} |",
        f"| p-value (two-sided) | {p_runs:.4f} |",
        f"| Reject H0 (p<0.05)? | {'YES — anomaly' if p_runs < 0.05 else 'No'} |",
        "",
        "---",
        "",
        "## 6. Summary",
        "",
        "| Test | Result | Interpretation |",
        "|------|--------|----------------|",
    ]

    # Summarize chi-square results
    any_position_reject = any(r["reject_h0_at_005"] for r in chi_results)
    positions_rejected = [str(r["position"] + 1) for r in chi_results if r["reject_h0_at_005"]]

    report_lines += [
        f"| Digit position χ² (6 positions) | {'REJECT H0 at pos ' + ', '.join(positions_rejected) if any_position_reject else 'Fail to reject H0'} | {'Possible mechanical bias — investigate in Phase 2' if any_position_reject else 'Looks uniform (expected)'} |",
        f"| Two-digit suffix χ² | {'REJECT H0' if two_p < 0.05 else 'Fail to reject H0'} | {'Suffix distribution anomaly' if two_p < 0.05 else 'Suffix looks uniform (expected)'} |",
        f"| Runs test | {'REJECT H0' if p_runs < 0.05 else 'Fail to reject H0'} | {'Serial dependence detected' if p_runs < 0.05 else 'No serial pattern (expected)'} |",
        "",
        "**Note:** Multiple comparisons problem applies across all tests above. No single",
        "rejection here constitutes evidence of edge. BH-FDR correction (SPEC §7.2) is",
        "required before any statistical claim. This EDA is exploratory only.",
        "",
        "---",
        "",
        "_Report generated by scripts/eda.py — Project Fortuna Phase 1_",
    ]

    EDA_REPORT.parent.mkdir(parents=True, exist_ok=True)
    EDA_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nEDA report written to: {EDA_REPORT}")
    print("\nPhase 1 EDA complete.")


if __name__ == "__main__":
    run_eda()
