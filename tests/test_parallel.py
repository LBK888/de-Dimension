"""Multi-core work must never be the reason a run fails.

This module exists because of one bug.  The desktop launcher used
`pythonw.exe` so that no console window sat behind the app; loky worker
processes cannot start under `pythonw.exe` on Windows, so every analysis died
part-way through the permutation test with a ``TerminatedWorkerError`` whose
message blames segmentation faults and memory -- neither of which had happened.
The expensive part of the run was already finished by then.

Two things are pinned here: the fallback returns the *same answer* as the
parallel path, and a genuine error inside the work still propagates instead of
being swallowed by a retry.
"""

from __future__ import annotations

import sys

import pytest

from somtrack.parallel import cpu_budget, hide_console, parallel_map


def _square(x):
    return x * x


def _explode(x):
    raise ValueError(f"deliberate failure on {x}")


# ==========================================================================
def test_parallel_map_matches_a_plain_loop():
    items = list(range(50))
    expected = [_square(x) for x in items]
    assert parallel_map(_square, items, n_jobs=2) == expected
    assert parallel_map(_square, items, n_jobs=1) == expected
    assert parallel_map(_square, items, n_jobs=2, prefer="threads") == expected


def test_parallel_map_handles_empty_and_single_inputs():
    assert parallel_map(_square, []) == []
    assert parallel_map(_square, [7]) == [49]


def test_a_real_error_is_not_swallowed_by_the_fallback():
    """A bug in the work must surface, not be retried into a wrong answer."""
    with pytest.raises(ValueError, match="deliberate failure"):
        parallel_map(_explode, [1, 2, 3], n_jobs=2)


def test_fallback_is_used_when_workers_cannot_start(monkeypatch):
    """Simulate the pythonw failure and check the answer still comes back."""
    from joblib.externals.loky.process_executor import TerminatedWorkerError

    import somtrack.parallel as par

    calls = {"n": 0}
    real = par.__dict__.get("_serial")

    def fake_parallel(*args, **kwargs):
        calls["n"] += 1
        if kwargs.get("prefer") == "processes":
            def run(_):
                raise TerminatedWorkerError("worker died on startup")
            return run
        from joblib import Parallel

        return Parallel(*args, **kwargs)

    monkeypatch.setattr("joblib.Parallel", fake_parallel)

    with pytest.warns(UserWarning, match="worker processes"):
        out = parallel_map(_square, list(range(20)), n_jobs=2)

    assert out == [x * x for x in range(20)], \
        "the fallback must produce the same answer, not a degraded one"
    assert calls["n"] >= 2, "it should have tried processes, then threads"
    assert real is not None


def test_cpu_budget_is_sane():
    assert cpu_budget(1) == 1
    assert cpu_budget(4) == 4
    assert cpu_budget(-1) >= 1
    assert cpu_budget(None) >= 1


# ==========================================================================
@pytest.mark.skipif(sys.platform != "win32", reason="Windows console only")
def test_hide_console_refuses_a_shared_console():
    """It must never hide a console window that belongs to the user's shell."""
    # Under pytest the console is shared with the shell that started it, so the
    # guard has to decline.  A wrong answer here would make somebody's terminal
    # vanish.
    assert hide_console() is False


def test_ensure_spawnable_is_idempotent_and_quiet_on_a_console():
    import somtrack.parallel as par

    par._FIXED = False
    try:
        first = par.ensure_spawnable()
        second = par.ensure_spawnable()
        assert second is None, "calling it twice must not repeat the work"
        if "pythonw" not in sys.executable.lower():
            assert first is None, "nothing to fix under a console interpreter"
    finally:
        par._FIXED = True
