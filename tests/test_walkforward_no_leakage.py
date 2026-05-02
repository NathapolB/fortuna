"""Walk-forward CV leakage tests — Phase 2 stub. SPEC §7.3.

These tests verify two independent assertions:
  (a) The target draw is NOT in the training set.
  (b) Every feature used to predict the target was computed strictly
      BEFORE draw_cutoff(target) = 06:00 Asia/Bangkok on draw_date.

Stubbed with @pytest.mark.skip(reason='Phase 2') per SPEC Phase 1 criterion 6.
Full implementation requires Phase 2 models + features.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 2")
def test_no_leakage_target_not_in_training():
    """Assert ctx.target_draw_id not in {d.draw_id for d in ctx.draws}.

    SPEC §7.3 Assertion 1.
    """
    raise NotImplementedError("Phase 2")


@pytest.mark.skip(reason="Phase 2")
def test_no_leakage_feature_timestamps():
    """Assert every feature computed_at < draw_cutoff(target_draw_id).

    SPEC §7.3 Assertion 2.
    draw_cutoff = 06:00 Asia/Bangkok on the draw_date.
    """
    raise NotImplementedError("Phase 2")


@pytest.mark.skip(reason="Phase 2")
def test_walk_forward_cv_window():
    """Full walk-forward CV: train on draws[:i], predict draws[i], repeat.

    SPEC §7.3 full test. Minimum train window = MIN_TRAIN draws.
    """
    raise NotImplementedError("Phase 2")
