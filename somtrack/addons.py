"""PyTorch for the Windows executable, downloaded the first time it is wanted.

SLISEMAP needs PyTorch, and PyTorch with GPU support is about 3 GB -- more than
a GitHub release file may hold, and far more than anyone without an NVIDIA
card should have to download.  So the full executable carries SLISEMAP but not
PyTorch, and fetches the build that suits *this* machine when it is asked to:

1. :func:`plan` reads the GPU and the driver (:mod:`somtrack.gpu`), and picks
   the newest PyTorch on download.pytorch.org that the driver can run and that
   exists for the Python inside the executable -- or the CPU build.
2. :func:`install_planned` downloads the wheel (resuming a broken download,
   checking its SHA-256), unpacks it into
   ``%LOCALAPPDATA%\\SOMTrack\\addons``, and then *checks it in a separate
   process*: a GPU build is only accepted once it has actually allocated on the
   GPU.  If it cannot, the CPU build is fetched instead, because a working CPU
   build beats a broken CUDA one.
3. :func:`activate`, called at start-up, puts the accepted build on
   ``sys.path``.

A wheel is a zip archive, so no pip is needed: there is none inside the
executable.  A source installation keeps using pip, through
``tools/install_torch.py``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import gpu
from .i18n import Text

ACTIVE = "active_torch.json"
CHECK_FLAG = "--somtrack-check-torch"
_CHUNK = 1 << 20

Progress = Callable[[str, float], None]


class Cancelled(Exception):
    """The download was stopped on request."""


# ==========================================================================
# Where it lives, and putting it to use
# ==========================================================================
def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def supported() -> bool:
    """Only the Windows executable downloads PyTorch itself."""
    return frozen() and sys.platform == "win32"


def home() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(base) / "SOMTrack" / "addons"


def installed() -> dict | None:
    """The accepted PyTorch download, or None."""
    try:
        info = json.loads((home() / ACTIVE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (home() / str(info.get("dir", "")) / "torch" / "__init__.py").exists():
        return None
    return info


def activate(clean: bool = False) -> dict | None:
    """Put the downloaded PyTorch on ``sys.path``.

    ``clean`` also removes builds that were replaced, and downloads that were
    abandoned; only the main process does that, never a worker.
    """
    info = installed()
    if info is not None:
        path = str(home() / info["dir"])
        if path not in sys.path:
            sys.path.append(path)
            importlib.invalidate_caches()
    if clean:
        _clean_up(keep=info["dir"] if info else None)
    return info


def _clean_up(keep: str | None) -> None:
    root = home()
    if not root.is_dir():
        return
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("torch-") and p.name != keep:
            shutil.rmtree(p, ignore_errors=True)
        elif p.name.endswith(".partial") and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


# ==========================================================================
# Choosing the build
# ==========================================================================
@dataclass
class Plan:
    channel: str                    # "cpu", or a CUDA channel such as "cu132"
    version: str                    # "2.15.0+cu132"
    url: str
    sha256: str | None
    size: int | None
    reason: str                     # why this build, in a sentence
    gpu: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def cuda(self) -> str:
        """"13.2" for cu132, "" for the CPU build."""
        m = re.fullmatch(r"cu(\d+)(\d)", self.channel)
        return f"{m.group(1)}.{m.group(2)}" if m else ""


def python_tag() -> str:
    return f"cp{sys.version_info[0]}{sys.version_info[1]}"


_WHEEL = re.compile(
    r'href="([^"#]*?torch-([^"/#]+?)-(cp\d+)-(cp\d+t?)-win_amd64\.whl)'
    r'(?:#sha256=([0-9a-fA-F]{64}))?"')


def wheels_in(html: str, index_url: str, tag: str) -> list[tuple[str, str, str | None]]:
    """``(version, url, sha256)`` of every release wheel for ``tag`` in an index page."""
    out = []
    for href, ver, py, abi, sha in _WHEEL.findall(html):
        ver = urllib.parse.unquote(ver)
        public = ver.split("+")[0]
        if py != tag or abi != tag or re.search(r"[a-z]", public):
            continue                    # another Python, free-threaded, or a pre-release
        out.append((ver, urllib.parse.urljoin(index_url, href), sha.lower() or None))
    return out


def _version_key(ver: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", ver.split("+")[0]))


def _request(url: str, method: str = "GET", headers: dict | None = None):
    req = urllib.request.Request(url, method=method,
                                 headers={"User-Agent": "SOMTrack", **(headers or {})})
    return urllib.request.urlopen(req, timeout=60)


def _size_of(url: str) -> int | None:
    try:
        with _request(url, "HEAD") as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except (OSError, ValueError):
        return None


def plan(cpu_only: bool = False) -> Plan:
    """The PyTorch build to download for this machine."""
    name, driver = gpu.detect_gpu()
    cap = gpu.compute_capability() if name else None
    if cpu_only:
        channels, why = [], Text("the CPU build was asked for")
    else:
        channels, why = gpu.usable_channels(name, driver, cap)

    tag, notes, error = python_tag(), [], None
    for ch in channels + ["cpu"]:
        index = f"{gpu.INDEX}/{ch}/torch/"
        try:
            with _request(index) as r:
                html = r.read().decode("utf-8", "replace")
        except OSError as exc:
            error = exc
            continue
        found = wheels_in(html, index, tag)
        if not found:
            notes.append(Text("{channel} has no PyTorch for Python {python}",
                              channel=ch, python=f"{sys.version_info[0]}."
                                                 f"{sys.version_info[1]}"))
            continue
        ver, url, sha = max(found, key=lambda w: _version_key(w[0]))
        reason = why
        if ch == "cpu" and channels:
            reason = Text("{why}, but no GPU build of PyTorch exists for this "
                          "Python, so the CPU build", why=why)
        return Plan(ch, ver, url, sha, _size_of(url), reason, name, notes)

    raise RuntimeError(Text("PyTorch could not be found on download.pytorch.org "
                            "({error}). Check the internet connection and try "
                            "again.", error=str(error or "no build for this Python")))


def human_size(n: int | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# ==========================================================================
# Downloading, unpacking, checking
# ==========================================================================
def _download(p: Plan, dest: Path, progress: Progress | None,
              cancelled: Callable[[], bool]) -> None:
    """Download with resume: a broken 3 GB download should not start again."""
    digest = hashlib.sha256()
    have = dest.stat().st_size if dest.exists() else 0
    if have:
        with open(dest, "rb") as f:
            for block in iter(lambda: f.read(_CHUNK), b""):
                digest.update(block)
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        r = _request(p.url, headers=headers)
    except urllib.error.HTTPError as exc:
        if exc.code != 416:                                 # 416: already complete
            raise
        r = None
    if r is not None:
        with r:
            if have and r.status != 206:                    # no resume: start over
                have, digest = 0, hashlib.sha256()
            total = p.size or (have + int(r.headers.get("Content-Length") or 0))
            with open(dest, "ab" if have else "wb") as f:
                done = have
                while True:
                    if cancelled():
                        raise Cancelled()
                    block = r.read(_CHUNK)
                    if not block:
                        break
                    f.write(block)
                    digest.update(block)
                    done += len(block)
                    if progress and total:
                        progress(Text("Downloading PyTorch ({done} of {total})",
                                      done=human_size(done), total=human_size(total)),
                                 0.85 * done / total)
    if p.sha256 and digest.hexdigest() != p.sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(Text("The PyTorch download was damaged (its checksum "
                                "does not match). Please try again."))


def _unpack(wheel: Path, target: Path, progress: Progress | None,
            cancelled: Callable[[], bool]) -> None:
    with zipfile.ZipFile(wheel) as z:
        members = z.infolist()
        total = sum(m.file_size for m in members) or 1
        done = 0
        for m in members:
            if cancelled():
                raise Cancelled()
            z.extract(m, target)
            done += m.file_size
            if progress:
                progress(Text("Unpacking PyTorch"), 0.85 + 0.1 * done / total)


def check(path: Path) -> dict:
    """Import PyTorch from ``path`` in a fresh process and, if it has CUDA,
    actually use the GPU.  A CUDA build that imports but cannot run a kernel
    (an RTX 50-series card on a pre-12.8 build) is caught here, not later."""
    if frozen():
        cmd = [sys.executable, CHECK_FLAG, str(path)]
    else:
        cmd = [sys.executable, "-m", "somtrack.addons", CHECK_FLAG, str(path)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                           creationflags=flags)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "checking PyTorch timed out"}
    for line in (p.stdout or "").splitlines():
        if line.startswith("SOMTRACK-TORCH "):
            return json.loads(line[len("SOMTRACK-TORCH "):])
    return {"ok": False, "error": (p.stderr or "no answer").strip()[-300:]}


def _check_main(path: str) -> None:
    """The other end of :func:`check`, run in the child process."""
    sys.path.insert(0, path)
    out: dict = {"ok": False}
    try:
        import torch

        out.update(ok=True, version=torch.__version__,
                   cuda=bool(torch.cuda.is_available()))
        if out["cuda"]:
            out["device"] = torch.cuda.get_device_name(0)
            try:
                (torch.zeros(8, device="cuda") + 1).sum().item()
                out["gpu_ok"] = True
            except Exception as exc:                       # no kernel image, ...
                out["gpu_ok"] = False
                out["error"] = str(exc)[:300]
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    print("SOMTRACK-TORCH " + json.dumps(out), flush=True)


def _install(p: Plan, progress: Progress | None,
             cancelled: Callable[[], bool]) -> tuple[str, dict]:
    root = home()
    root.mkdir(parents=True, exist_ok=True)
    need = 3 * (p.size or 3 * 1024 ** 3)              # wheel + unpacked + slack
    if shutil.disk_usage(root).free < need:
        raise RuntimeError(Text("Not enough free disk space for PyTorch: about "
                                "{need} is needed on the drive holding {path}.",
                                need=human_size(need), path=str(root)))
    name = "torch-" + re.sub(r"[^0-9A-Za-z.]+", "-", p.version)
    wheel = root / f"{name}.whl.download"
    _download(p, wheel, progress, cancelled)

    staging = root / f"{name}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    _unpack(wheel, staging, progress, cancelled)
    wheel.unlink(missing_ok=True)
    target = root / name
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)

    if progress:
        progress(Text("Checking PyTorch"), 0.97)
    return name, check(target)


def install_planned(p: Plan, progress: Progress | None = None,
                    cancelled: Callable[[], bool] = lambda: False) -> dict:
    """Download ``p``, check it, fall back to the CPU build, and accept one."""
    notes = list(p.notes)
    name, result = _install(p, progress, cancelled)
    if p.channel != "cpu" and not (result.get("ok") and result.get("gpu_ok")):
        notes.append(Text("The GPU build of PyTorch could not use the GPU ({error}); "
                          "the CPU build is used instead.",
                          error=result.get("error") or "CUDA is not available"))
        cpu = plan(cpu_only=True)
        name, result = _install(cpu, progress, cancelled)
        p = cpu
    if not result.get("ok"):
        raise RuntimeError(Text("PyTorch was downloaded but does not start: {error}",
                                error=result.get("error") or "unknown error"))

    info = {"dir": name, "version": result.get("version", p.version),
            "channel": p.channel, "cuda": p.cuda,
            "device": result.get("device") if result.get("gpu_ok") else "",
            "plan": {k: v for k, v in asdict(p).items() if k != "notes"}}
    (home() / ACTIVE).write_text(json.dumps(info, indent=2, default=str),
                                 encoding="utf-8")
    activate(clean=True)
    info["notes"] = notes
    if progress:
        progress(Text("PyTorch is ready"), 1.0)
    return info


__all__ = ["Plan", "Cancelled", "supported", "home", "installed", "activate",
           "plan", "install_planned", "check", "human_size", "wheels_in",
           "CHECK_FLAG"]


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == CHECK_FLAG:
    _check_main(sys.argv[2])
