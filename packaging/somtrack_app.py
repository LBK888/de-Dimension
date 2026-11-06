"""Entry point of the Windows executable built by PyInstaller.

Run with no arguments it opens the desktop app; with arguments it is the
command line, exactly like ``python -m somtrack``.  Two things have to be
settled before any of SOMTrack is imported.

**Worker processes start this same .exe.**  joblib's loky backend launches a
worker as ``SOMTrack.exe --multiprocessing-fork <handle>`` and its resource
tracker as ``SOMTrack.exe -c "from joblib.externals.loky... import main; ..."``;
the standard library uses ``--multiprocessing-fork key=value ...``.  Each of
those has to do its job and exit.  Falling through to the normal start-up
would open one more SOMTrack window per worker.

**numba caches compiled code next to the source**, which inside the bundle is
not writable, so UMAP and PaCMAP would fail on import.  The cache goes to
``%LOCALAPPDATA%\\SOMTrack\\numba_cache`` instead.
"""

from __future__ import annotations

import os
import sys
import tempfile


def _set_up_environment() -> None:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    cache = os.path.join(base, "SOMTrack", "numba_cache")
    os.makedirs(cache, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", cache)


def _run_child_process() -> bool:
    """Run a worker / tracker process and return True, or return False."""
    argv = sys.argv
    if len(argv) >= 2 and argv[1] == "--multiprocessing-fork":
        if len(argv) == 3 and "=" not in argv[2]:            # a loky worker
            from joblib.externals.loky.backend.popen_loky_win32 import main

            # In a frozen program loky passes only the pipe handle.  That
            # handle belongs to the parent, so without the parent's pid the
            # worker cannot duplicate it and dies at once ("WinError 6: the
            # handle is invalid"), taking scikit-learn's n_jobs=-1 with it.
            main(int(argv[2]), parent_pid=os.getppid())       # exits itself
        else:                                                 # the standard library
            import multiprocessing

            multiprocessing.freeze_support()
        return True
    if len(argv) >= 3 and argv[-2] == "-c" and argv[-1].startswith(
            ("from joblib.externals.loky", "from multiprocessing")):
        exec(argv[-1], {"__name__": "__main__"})              # resource tracker
        return True
    return False


def main() -> int:
    _set_up_environment()
    if _run_child_process():
        return 0
    from somtrack.__main__ import main as somtrack_main

    return somtrack_main()


if __name__ == "__main__":
    sys.exit(main())
