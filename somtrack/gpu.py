"""Which PyTorch build this machine can actually run.

Shared by ``tools/install_torch.py`` (the source installation) and
:mod:`somtrack.addons` (the Windows executable, which downloads PyTorch the
first time SLISEMAP is wanted).  The decision is a real one, because getting it
wrong fails at *run* time rather than at install time:

* A GeForce RTX 50-series card is Blackwell, compute capability 12.0.  PyTorch
  builds before CUDA 12.8 contain no kernels for it, so they install cleanly,
  import cleanly, and then raise "no kernel image is available for execution on
  the device" the first time anything touches the GPU.
* A CUDA build newer than the installed driver supports fails the same way.
  ``nvidia-smi`` reports the highest CUDA version the driver can run, so the
  channel is chosen against that rather than assumed.
"""

from __future__ import annotations

import re
import subprocess

from .i18n import Text

# PyTorch CUDA channels, newest first.  Each entry is the channel name and the
# CUDA version it needs the driver to support.
CHANNELS: list[tuple[str, tuple[int, int]]] = [
    ("cu132", (13, 2)),
    ("cu130", (13, 0)),
    ("cu129", (12, 9)),
    ("cu128", (12, 8)),
    ("cu126", (12, 6)),
    ("cu124", (12, 4)),
]

# Blackwell (RTX 50-series, compute capability 12.0) has no kernels in any
# PyTorch built before CUDA 12.8.
MIN_CUDA_FOR_BLACKWELL = (12, 8)

INDEX = "https://download.pytorch.org/whl"
CPU_INDEX = f"{INDEX}/cpu"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    # CREATE_NO_WINDOW: from the desktop app, nvidia-smi must not flash a console
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(cmd, capture_output=True, text=True, creationflags=flags,
                          **kw)


# "CUDA Version: 12.8" on older drivers, "CUDA UMD Version: 13.3" from the
# 600 series on.  Missing the new spelling made a working RTX 5090 look like
# a card whose driver could not be read, and it got the CPU build.
_DRIVER_CUDA = re.compile(r"CUDA(?: UMD| Driver)? Version:\s*(\d+)\.(\d+)")


def detect_gpu() -> tuple[str | None, tuple[int, int] | None]:
    """Return (GPU name, highest CUDA version the driver supports)."""
    try:
        p = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                 timeout=30)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None, None
    if p.returncode != 0 or not p.stdout.strip():
        return None, None
    name = p.stdout.strip().splitlines()[0].strip()

    driver_cuda = None
    try:
        q = _run(["nvidia-smi"], timeout=30)
        m = _DRIVER_CUDA.search(q.stdout or "")
        if m:
            driver_cuda = (int(m.group(1)), int(m.group(2)))
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return name, driver_cuda


def compute_capability() -> tuple[int, int] | None:
    """The first GPU's compute capability, e.g. (12, 0), if the driver says."""
    try:
        p = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                 timeout=30)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    m = re.match(r"\s*(\d+)\.(\d+)", p.stdout or "") if p.returncode == 0 else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def is_blackwell_or_newer(name: str,
                          capability: tuple[int, int] | None = None) -> bool:
    """RTX 50-series and the matching workstation cards (compute 12.0+)."""
    if capability is not None:
        return capability >= (12, 0)
    n = name.upper()
    if re.search(r"RTX\s*5\d{3}", n):
        return True
    return any(k in n for k in ("BLACKWELL", "RTX PRO 6000", "B200", "GB200"))


def usable_channels(name: str | None,
                    driver_cuda: tuple[int, int] | None,
                    capability: tuple[int, int] | None = None
                    ) -> tuple[list[str], str]:
    """Every CUDA channel this GPU and driver can run, newest first, and why.

    An empty list means the CPU build.  More than one is returned because a
    channel may have no build for this Python version; the caller tries them
    in order.
    """
    if name is None:
        return [], Text("no NVIDIA GPU was found")

    # Only Blackwell has a floor.  Older cards run happily on older CUDA, and
    # telling the owner of a working RTX 4090 to update their driver because
    # SOMTrack only looked at the newest channels would be nonsense.
    blackwell = is_blackwell_or_newer(name, capability)
    needs = (MIN_CUDA_FOR_BLACKWELL if blackwell
             else min(cuda for _, cuda in CHANNELS))

    if driver_cuda is None:
        return [], Text("{name} was found but the driver's CUDA version could "
                        "not be read", name=name)

    ok = [ch for ch, cuda in CHANNELS if needs <= cuda <= driver_cuda]
    if ok:
        return ok, Text("{name}, driver supports CUDA {cuda}", name=name,
                        cuda=f"{driver_cuda[0]}.{driver_cuda[1]}")

    if driver_cuda < needs:
        template = ("{name} is a Blackwell card and needs CUDA {needs} or newer, "
                    "but the driver only supports {have} -- updating the NVIDIA "
                    "driver would enable GPU support" if blackwell else
                    "{name} needs CUDA {needs} or newer, but the driver only "
                    "supports {have} -- updating the NVIDIA driver would enable "
                    "GPU support")
        return [], Text(template, name=name, needs=f"{needs[0]}.{needs[1]}",
                        have=f"{driver_cuda[0]}.{driver_cuda[1]}")
    return [], Text("{name}, but no matching PyTorch build is published",
                    name=name)


def choose_channel(name: str | None,
                   driver_cuda: tuple[int, int] | None,
                   capability: tuple[int, int] | None = None
                   ) -> tuple[str | None, str]:
    """The newest usable CUDA channel, or None for the CPU build, and why."""
    channels, why = usable_channels(name, driver_cuda, capability)
    return (channels[0] if channels else None), why


__all__ = ["CHANNELS", "MIN_CUDA_FOR_BLACKWELL", "INDEX", "CPU_INDEX",
           "detect_gpu", "compute_capability", "is_blackwell_or_newer",
           "usable_channels",
           "choose_channel"]
