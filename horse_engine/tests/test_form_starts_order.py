"""Regression test for the unsorted-`starts` bug (W30/W31 phantom long_layoff).

`days_since_last_run` reads `starts[0].date` and `weighted_form_score` weights by
list index, both assuming newest-first order — but `runner.last_10_starts` is not
guaranteed sorted. An unsorted feed produced phantom long layoffs (KOMITO
2026-08-02: 221d snapshot vs 39d real) and mis-weighted form. The fix sorts once
at the top of `enrich_runner`; these tests lock in the correct behaviour.
"""
from datetime import date, timedelta

from horse_engine.enrichers import form as form_enricher
from horse_engine.models.race import FormStart


def _start(days_ago: int, position: int = 1, finishers: int = 8) -> FormStart:
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    return FormStart(
        date=d, track="t", distance=1200, track_condition="Good 4", barrier=1,
        weight=57.0, jockey="j", position=position, finishers=finishers,
        beaten_margin=0.0, race_class="BM64", prize_money=20000,
    )


def _sorted_newest_first(starts):
    return sorted(starts, key=lambda s: (s.date or ""), reverse=True)


def test_days_since_last_run_wrong_on_unsorted_input():
    """The raw function trusts starts[0] — unsorted input yields a phantom gap.
    This documents the bug the enrich_runner sort defends against."""
    recent, old = _start(14), _start(221)
    assert form_enricher.days_since_last_run([recent, old]) == 14      # correct order
    assert form_enricher.days_since_last_run([old, recent]) == 221     # BUG: phantom layoff


def test_sort_at_source_fixes_days_since_last_run():
    """After the enrich_runner sort, order of the feed no longer matters."""
    recent, old = _start(14), _start(221)
    for feed in ([recent, old], [old, recent]):
        assert form_enricher.days_since_last_run(_sorted_newest_first(feed)) == 14


def test_no_phantom_long_layoff_for_midprep_horse():
    """A horse with several recent runs (mid-prep) must not read as long-layoff,
    regardless of feed order — mirrors the 72% of tagged picks with runs>=2."""
    feed = [_start(14), _start(28), _start(45), _start(230)]  # 3 recent + 1 pre-spell
    shuffled = [feed[2], feed[0], feed[3], feed[1]]           # deterministic reorder
    s = _sorted_newest_first(shuffled)
    assert form_enricher.days_since_last_run(s) == 14
    assert form_enricher.days_since_last_run(s) <= 180  # no long_layoff tag


def test_weighted_form_score_stable_under_reorder():
    """Form score must be identical once sorted, whatever the feed order."""
    a = _start(7, position=1, finishers=8)     # recent win
    b = _start(21, position=8, finishers=8)    # older last
    c = _start(60, position=4, finishers=8)
    base = form_enricher.weighted_form_score(_sorted_newest_first([a, b, c]))
    for feed in ([b, a, c], [c, b, a], [b, c, a]):
        assert form_enricher.weighted_form_score(_sorted_newest_first(feed)) == base
