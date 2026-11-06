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
import subprocess
import sys
from pathlib import Path

# The GPU detection and the choice of channel are shared with the Windows
# executable, which downloads PyTorch itself (somtrack/addons.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from somtrack.gpu import (CPU_INDEX, choose_channel, compute_capability,  # noqa: E402
                          detect_gpu)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


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
        else choose_channel(name, driver_cuda, compute_capability())

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
