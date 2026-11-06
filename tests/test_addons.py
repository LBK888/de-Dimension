"""PyTorch on demand for the Windows executable, and the GPU checks behind it.

Nothing here touches the network: a fake PyTorch wheel, served from a file://
URL, goes through the real download, checksum, unpack and separate-process
check, including the fall-back from a GPU build that cannot use the GPU.
"""

from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from pathlib import Path

import pytest

from somtrack import addons, gpu


# ==========================================================================
# Which build this machine can run
# ==========================================================================
@pytest.mark.parametrize("line, want", [
    ("| NVIDIA-SMI 550.54   Driver Version: 550.54   CUDA Version: 12.4 |", (12, 4)),
    # the 600-series drivers renamed the field; missing it gave an RTX 5090
    # the CPU build
    ("| NVIDIA-SMI 610.88  KMD Version: 610.88  CUDA UMD Version: 13.3 |", (13, 3)),
])
def test_both_spellings_of_the_driver_cuda_version_are_read(line, want):
    m = gpu._DRIVER_CUDA.search(line)
    assert m and (int(m.group(1)), int(m.group(2))) == want


def test_a_blackwell_card_never_gets_a_build_without_its_kernels():
    channels, _ = gpu.usable_channels("NVIDIA GeForce RTX 5090", (13, 3), (12, 0))
    assert channels and channels[0] == "cu132"
    assert all(dict(gpu.CHANNELS)[c] >= gpu.MIN_CUDA_FOR_BLACKWELL for c in channels)
    # identified by compute capability even when the name does not give it away
    channels, why = gpu.usable_channels("NVIDIA Graphics Device", (12, 6), (12, 0))
    assert channels == [] and "12.8" in why


def test_an_older_card_is_not_told_to_update_a_working_driver():
    channels, _ = gpu.usable_channels("NVIDIA GeForce RTX 4090", (12, 4), (8, 9))
    assert channels == ["cu124"]


def test_no_gpu_means_the_cpu_build():
    assert gpu.choose_channel(None, None)[0] is None


# ==========================================================================
# Reading the download index
# ==========================================================================
def test_the_index_is_read_for_this_python_only():
    sha = "ab" * 32
    html = "".join(f'<a href="/whl/cu132/torch-{v}-{t}-win_amd64.whl{h}">x</a>' for v, t, h in (
        ("2.15.0%2Bcu132", "cp314-cp314", f"#sha256={sha}"),
        ("2.15.0%2Bcu132", "cp314-cp314t", ""),          # free-threaded
        ("2.16.0.dev20260101%2Bcu132", "cp314-cp314", ""),  # pre-release
        ("2.15.0%2Bcu132", "cp313-cp313", ""),           # another Python
        ("2.9.1%2Bcu132", "cp314-cp314", ""),
    ))
    found = addons.wheels_in(html, "https://download.pytorch.org/whl/cu132/torch/",
                             "cp314")
    assert [v for v, _, _ in found] == ["2.15.0+cu132", "2.9.1+cu132"]
    assert found[0][2] == sha
    assert found[0][1].startswith("https://download.pytorch.org/whl/cu132/torch-2.15.0")
    assert max(found, key=lambda w: addons._version_key(w[0]))[0] == "2.15.0+cu132"


def test_plan_reports_the_cuda_version_of_its_channel():
    p = addons.Plan("cu132", "2.15.0+cu132", "u", None, None, "r")
    assert p.cuda == "13.2"
    assert addons.Plan("cpu", "2.15.0+cpu", "u", None, None, "r").cuda == ""


# ==========================================================================
# Download, unpack, check, accept
# ==========================================================================
def _fake_torch(tmp_path: Path, name: str, gpu_works: bool | None) -> tuple[str, str, int]:
    """A wheel holding a stand-in 'torch'.  gpu_works None = CPU build."""
    cuda = "False" if gpu_works is None else "True"
    zeros = ("    return _T()\n" if gpu_works in (None, True) else
             "    raise RuntimeError('no kernel image is available')\n")
    code = (
        "__version__ = '9.9.9'\n"
        "class _T:\n"
        "    def __add__(self, o): return self\n"
        "    def sum(self): return self\n"
        "    def item(self): return 8.0\n"
        "def zeros(*a, **k):\n" + zeros +
        "class cuda:\n"
        f"    is_available = staticmethod(lambda: {cuda})\n"
        "    get_device_name = staticmethod(lambda i: 'Fake GPU')\n")
    wheel = tmp_path / name
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("torch/__init__.py", code)
        z.writestr("torch-9.9.9.dist-info/METADATA", "Name: torch\n")
    data = wheel.read_bytes()
    return wheel.as_uri(), hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture
def addon_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.chdir(Path(__file__).resolve().parents[1])   # for -m somtrack.addons
    saved = list(sys.path)
    yield tmp_path
    sys.path[:] = saved


def test_a_cpu_build_is_downloaded_checked_and_activated(addon_home):
    url, sha, size = _fake_torch(addon_home, "cpu.whl", gpu_works=None)
    p = addons.Plan("cpu", "9.9.9+cpu", url, sha, size, "test")
    seen = []
    info = addons.install_planned(p, progress=lambda m, f: seen.append(f))
    assert info["version"] == "9.9.9" and info["channel"] == "cpu"
    assert seen and seen[-1] == 1.0
    assert addons.installed()["dir"] == info["dir"]
    assert str(addons.home() / info["dir"]) in sys.path


def test_a_gpu_build_that_cannot_use_the_gpu_is_replaced_by_the_cpu_build(
        addon_home, monkeypatch):
    bad = addons.Plan("cu132", "9.9.9+cu132",
                      *_fake_torch(addon_home, "gpu.whl", gpu_works=False), "test")
    cpu_url, cpu_sha, cpu_size = _fake_torch(addon_home, "cpu.whl", gpu_works=None)
    monkeypatch.setattr(addons, "plan", lambda cpu_only=False: addons.Plan(
        "cpu", "9.9.9+cpu", cpu_url, cpu_sha, cpu_size, "fallback"))
    info = addons.install_planned(bad)
    assert info["channel"] == "cpu"
    assert any("could not use the GPU" in n for n in info["notes"])


def test_a_working_gpu_build_is_accepted_with_its_device(addon_home):
    p = addons.Plan("cu132", "9.9.9+cu132",
                    *_fake_torch(addon_home, "gpu.whl", gpu_works=True), "test")
    info = addons.install_planned(p)
    assert info["channel"] == "cu132" and info["device"] == "Fake GPU"


def test_a_damaged_download_is_refused(addon_home):
    url, _, size = _fake_torch(addon_home, "cpu.whl", gpu_works=None)
    p = addons.Plan("cpu", "9.9.9+cpu", url, "0" * 64, size, "test")
    with pytest.raises(RuntimeError, match="checksum"):
        addons.install_planned(p)
    assert addons.installed() is None


def test_only_the_executable_downloads_pytorch_itself():
    assert not addons.supported()          # the test suite is not frozen


# ==========================================================================
# The Analysis page
# ==========================================================================
@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_slisemap_offers_the_download_and_becomes_tickable(qapp, monkeypatch):
    """In the full executable SLISEMAP is present but PyTorch is not yet."""
    from somtrack.analysis import get
    from somtrack.analysis.registry import MethodSpec
    from somtrack.ui.pages import AppState, ClusterPage

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(addons, "supported", lambda: True)
    monkeypatch.setattr(MethodSpec, "bundled", lambda self: True)
    spec = get("slisemap")
    original = spec.available
    object.__setattr__(spec, "available", lambda: False)
    try:
        page = ClusterPage(AppState())
        btn = page._torch_buttons["slisemap"]
        assert not btn.isHidden()
        assert "needs PyTorch" in page._boxes["slisemap"].text()
        assert not page._boxes["slisemap"].isEnabled()

        object.__setattr__(spec, "available", lambda: True)     # downloaded
        page.supervised_ack.setChecked(True)
        page.refresh_availability()
        assert btn.isHidden()
        assert page._boxes["slisemap"].text() == spec.label
        assert page._boxes["slisemap"].isEnabled()
    finally:
        object.__setattr__(spec, "available", original)
