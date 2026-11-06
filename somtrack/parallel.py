"""Running work across cores, without letting that be a way for a run to fail.

Two problems live here, both of them Windows problems, and both of them able to
destroy an analysis that had already finished the expensive part.

**`pythonw.exe` cannot start worker processes.**  The desktop app is launched
with `pythonw.exe` so that no console window sits behind it.  `joblib` starts
its workers with `sys.executable`, which is then `pythonw.exe`, and those
workers die the moment they start -- reported back as a
``TerminatedWorkerError`` whose message talks about segmentation faults and
memory, neither of which is what happened.  :func:`ensure_spawnable` points
multiprocessing at the `python.exe` sitting next to it instead.

**A permutation test is not worth losing to a parallelism problem.**  The
result of :func:`parallel_map` is the same however it was computed, so a failure
to start workers falls back to threads and then to a plain loop.  It is slower
and it says so; it does not throw away a p value the user waited two minutes
for.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Iterable, Sequence

_FIXED = False


# ==========================================================================
def ensure_spawnable() -> str | None:
    """Make sure worker processes can actually start.  Returns a note, or None.

    Idempotent and safe to call from anywhere; it does nothing at all off
    Windows, or when the interpreter can already spawn.

    **Under `pythonw.exe`, loky workers die and nothing fixes it.**  Pointing
    `sys.executable`, `multiprocessing` and loky's own `_python_exe` at the
    console interpreter does not help; nor does giving the process valid stdio
    handles.  All three were tried, and a worker still dies on startup, which
    surfaces as a ``TerminatedWorkerError`` whose message blames segmentation
    faults and memory -- neither of which is what happened.

    So the launcher runs `python.exe` and hides its console window instead (see
    :func:`hide_console`).  This function stays as the safety net for anyone who
    starts the app another way: it switches multi-core work off rather than
    letting scikit-learn's own ``n_jobs=-1`` end the run two minutes in.
    """
    global _FIXED
    if _FIXED:
        return None
    _FIXED = True

    if Path(sys.executable).name.lower() != "pythonw.exe":
        return None

    _force_single_core()
    return ("running under pythonw.exe, which cannot start worker processes, "
            "so the analysis is using one core")


def _force_single_core() -> None:
    """Stop joblib using processes anywhere, including inside scikit-learn."""
    os.environ["JOBLIB_MULTIPROCESSING"] = "0"
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    warnings.warn(
        "Worker processes cannot be started here, so multi-core work is "
        "switched off. Results are unchanged; the analysis takes longer. "
        "Running SOMTrack from a console with python.exe restores it.")


def hide_console() -> bool:
    """Hide this process's own console window, if it has one to itself.

    The desktop app runs on `python.exe` because `pythonw.exe` cannot start
    worker processes, and a console window sitting behind the app looks like a
    fault.  Hiding it keeps both: real multi-core work and no black box.

    Only a console this process owns alone is hidden.  Started from a terminal,
    the console belongs to the shell and hiding it would take the user's window
    away with it.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        window = k32.GetConsoleWindow()
        if not window:
            return False                       # no console: nothing to hide

        # GetConsoleProcessList returns how many processes share this console.
        # More than one means we are a guest in somebody else's window.
        buf = (wintypes.DWORD * 8)()
        count = k32.GetConsoleProcessList(buf, 8)
        if count != 1:
            return False

        ctypes.WinDLL("user32", use_last_error=True).ShowWindow(window, 0)  # SW_HIDE
        return True
    except Exception:
        return False


def cpu_budget(n_jobs: int = -1) -> int:
    """How many workers to actually ask for."""
    if n_jobs is not None and n_jobs > 0:
        return int(n_jobs)
    n = os.cpu_count() or 1
    if n_jobs in (None, -1):
        return max(1, n)
    return max(1, n + 1 + int(n_jobs))


# ==========================================================================
def parallel_map(
    fn: Callable,
    items: Sequence | Iterable,
    *,
    n_jobs: int = -1,
    prefer: str = "processes",
    desc: str = "this step",
    progress=None,
) -> list:
    """``[fn(x) for x in items]``, across cores where that is possible.

    Falls back through threads to a plain loop rather than failing.  The three
    routes produce identical results; only the time differs.
    """
    items = list(items)
    if not items:
        return []

    workers = cpu_budget(n_jobs)
    if workers <= 1 or len(items) == 1:
        return _serial(fn, items, progress)

    ensure_spawnable()

    from joblib import Parallel, delayed

    try:
        out = Parallel(n_jobs=workers, prefer=prefer)(
            delayed(fn)(x) for x in items)
        if progress:
            progress(len(items), len(items))
        return list(out)
    except Exception as exc:
        if not _is_worker_failure(exc):
            raise
        warnings.warn(
            f"Could not start worker processes for {desc} ({type(exc).__name__}), "
            f"so it ran on threads instead. The result is the same; it took "
            f"longer.")

    try:
        out = Parallel(n_jobs=workers, prefer="threads")(
            delayed(fn)(x) for x in items)
        if progress:
            progress(len(items), len(items))
        return list(out)
    except Exception as exc:
        warnings.warn(
            f"Parallel execution of {desc} failed entirely ({type(exc).__name__}); "
            f"running on a single core.")

    return _serial(fn, items, progress)


def _serial(fn: Callable, items: list, progress=None) -> list:
    out = []
    for i, x in enumerate(items):
        out.append(fn(x))
        if progress and (i % 16 == 0 or i == len(items) - 1):
            progress(i + 1, len(items))
    return out


def _is_worker_failure(exc: BaseException) -> bool:
    """Is this joblib failing to run, rather than the work itself failing?

    A genuine error inside ``fn`` must keep propagating -- retrying it on
    threads would only hide a real bug and produce a wrong answer quietly.
    """
    names = {type(exc).__name__}
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        names.add(type(cause).__name__)
    return bool(names & {
        "TerminatedWorkerError",      # worker died on startup (pythonw.exe)
        "BrokenProcessPool",
        "BrokenExecutor",
        "PicklingError",
        "PermissionError",            # locked-down or sandboxed machines
        "OSError",
    })


__all__ = ["ensure_spawnable", "parallel_map", "cpu_budget"]
