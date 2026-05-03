"""Statistical tests for "beat random" evaluation. SPEC §7.2. Phase 2."""

from __future__ import annotations


def binomial_test(hits: int, n: int, p_null: float) -> tuple[float, bool]:
    """Two-sided binomial test. SPEC §7.2.

    H0: true hit rate = p_null.
    Minimum n = 50 before any claim (SPEC §7.2).

    Returns (p_value, valid) where valid=False if n < 50.

    Uses scipy.stats.binomtest (scipy>=1.7) for forward-compatibility.
    """
    if n < 50:
        return 1.0, False

    try:
        from scipy.stats import binomtest  # type: ignore  # scipy >= 1.7
        result = binomtest(hits, n, p_null, alternative="two-sided")
        return result.pvalue, True
    except ImportError:
        # Fallback for older scipy
        from scipy.stats import binom_test  # type: ignore
        p_value = binom_test(hits, n, p_null, alternative="two-sided")
        return p_value, True


def bh_fdr_correction(
    p_values: list[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg FDR correction. SPEC §7.2.

    Returns (reject_flags, corrected_p_values).
    """
    from statsmodels.stats.multitest import multipletests  # type: ignore

    if not p_values:
        return [], []
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return list(reject), list(pvals_corrected)


def chi_square_uniformity(
    counts: list[int],
) -> tuple[float, float]:
    """Chi-square goodness-of-fit test for uniformity. SPEC §7.2.

    H0: all bins equally likely.
    Returns (statistic, p_value).
    """
    from scipy.stats import chisquare  # type: ignore

    stat, p_value = chisquare(counts)
    return float(stat), float(p_value)
