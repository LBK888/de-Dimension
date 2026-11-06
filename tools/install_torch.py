"""Install PyTorch, matched to whatever GPU and driver this machine has.

PyTorch is only needed for SLISEMAP, and it is the largest thing SOMTrack can
pull in -- a CUDA build is several gigabytes against about 250 MB for the CPU
one.  It is therefore installed by a step of its own that can be skipped.

The reason this is Python and not more batch script: choosing the right build
is a real decision, and getting it wrong fails at *run* time rather than at
install time.

* A GeForce RTX 50-series card is Blackwell, compute capability 12.0.  PyTorch
  builds before CUDA 12.8 contain no kernels for it, so they install cleanly,
  import cleanly, and then raise "no kernel image is available for execution on
  the device" the first time anything touches the GPU.
* A CUDA build newer than the installed driver supports fails the same way.
  `nvidia-smi` reports the highest CUDA version the driver can run, so the
  channel is chosen against that rather than assumed.

Both are checked here, and the install is verified by actually allocating on
the GPU before it is called a success.  If anything does not line up, the CPU
build goes in instead: SLISEMAP on a few hundred samples does not need a GPU,
and a working CPU build beats a broken CUDA one.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

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

CPU_INDEX = "https://download.pytorch.org/whl/cpu"


# ==========================================================================
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


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
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", q.stdout or "")
        if m:
            driver_cuda = (int(m.group(1)), int(m.group(2)))
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return name, driver_cuda


def is_blackwell_or_newer(name: str) -> bool:
    """RTX 50-series and the matching workstation cards."""
    n = name.upper()
    if re.search(r"RTX\s*5\d{3}", n):
        return True
    return any(k in n for k in ("BLACKWELL", "RTX PRO 6000", "B200", "GB200"))


def choose_channel(name: str | None,
                   driver_cuda: tuple[int, int] | None) -> tuple[str | None, str]:
    """Pick a CUDA channel, or None for the CPU build.  Returns (channel, why)."""
    if name is None:
        return None, "no NVIDIA GPU was found"

    # Only Blackwell has a floor.  Older cards run happily on older CUDA, and
    # telling the owner of a working RTX 4090 to update their driver because
    # SOMTrack only looked at the newest channels would be nonsense.
    if is_blackwell_or_newer(name):
        needs = MIN_CUDA_FOR_BLACKWELL
    else:
        needs = min(cuda for _, cuda in CHANNELS)

    if driver_cuda is None:
        return None, (f"{name} was found but the driver's CUDA version could "
                      f"not be read")

    for channel, cuda in CHANNELS:
        if cuda <= driver_cuda and cuda >= needs:
            return channel, (f"{name}, driver supports CUDA "
                             f"{driver_cuda[0]}.{driver_cuda[1]}")

    if driver_cuda < needs:
        reason = ("is a Blackwell card and needs CUDA "
                  f"{needs[0]}.{needs[1]} or newer" if is_blackwell_or_newer(name)
                  else f"needs CUDA {needs[0]}.{needs[1]} or newer")
        return None, (f"{name} {reason}, but the driver only supports "
                      f"{driver_cuda[0]}.{driver_cuda[1]} -- updating the "
                      f"NVIDIA driver would enable GPU support")
    return None, f"{name}, but no matching PyTorch build is published"


# ==========================================================================
def pip_install(args: list[str], quiet: bool = False) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"] + args
    if quiet:
        cmd.insert(4, "--quiet")
    print("      " + " ".join(cmd[3:]), flush=True)
    return subprocess.run(cmd).returncode == 0


def verify(expect_cuda: bool) -> tuple[bool, str]:
    """Import torch and, if a GPU build was requested, actually use the GPU."""
    code = (
        "import torch, sys;"
        "cuda = torch.cuda.is_available();"
        "name = torch.cuda.get_device_name(0) if cuda else '';"
        "ok = True\n"
        "if cuda:\n"
        "    try:\n"
        "        (torch.zeros(8, device='cuda') + 1).sum().item()\n"
        "    except Exception as e:\n"
        "        ok = False; name = str(e)[:120]\n"
        "print(f'{torch.__version__}|{int(cuda)}|{int(ok)}|{name}')\n"
    )
    p = _run([sys.executable, "-c", code], timeout=300)
    if p.returncode != 0:
        return False, (p.stderr or "torch does not import").strip()[-200:]

    version, cuda, ok, name = (p.stdout.strip().split("|") + ["", "", "", ""])[:4]
    if cuda == "1" and ok == "1":
        return True, f"torch {version} using {name}"
    if cuda == "1" and ok != "1":
        return False, f"torch {version} sees the GPU but cannot use it: {name}"
    if expect_cuda:
        return False, f"torch {version} installed, but without GPU support"
    return True, f"torch {version} (CPU)"


# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-only", action="store_true",
                    help="skip the GPU build even if a GPU is present")
    ap.add_argument("--no-slisemap", action="store_true",
                    help="install PyTorch but not SLISEMAP")
    args = ap.parse_args(argv)

    name, driver_cuda = detect_gpu()
    channel, why = (None, "asked for the CPU build") if args.cpu_only \
        else choose_channel(name, driver_cuda)

    print(f"      {why}")
    if channel:
        print(f"      installing the CUDA build ({channel}); this is a large "
              f"download, around 3 GB")
        ok = pip_install(["torch", "--index-url",
                          f"https://download.pytorch.org/whl/{channel}"])
    else:
        print("      installing the CPU build, about 250 MB. SLISEMAP on a few "
              "hundred samples does not need a GPU.")
        ok = pip_install(["torch", "--index-url", CPU_INDEX])

    if not ok and channel:
        print("      the CUDA build failed; falling back to the CPU build")
        channel = None
        ok = pip_install(["torch", "--index-url", CPU_INDEX])

    if not ok:
        print("  [!] PyTorch did not install. SLISEMAP will not be available; "
              "everything else works.")
        return 1

    good, note = verify(expect_cuda=bool(channel))
    print(f"      {note}")
    if not good and channel:
        print("      reinstalling the CPU build, which will work")
        pip_install(["torch", "--index-url", CPU_INDEX,
                     "--force-reinstall", "--no-deps"])
        good, note = verify(expect_cuda=False)
        print(f"      {note}")

    if not good:
        print("  [!] PyTorch is installed but not usable. SLISEMAP will be "
              "skipped; everything else works.")
        return 1

    if not args.no_slisemap:
        print("      - slisemap")
        if not pip_install(["slisemap"], quiet=True):
            print("  [!] SLISEMAP did not install; everything else works.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
