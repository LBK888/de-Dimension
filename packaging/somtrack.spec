# PyInstaller spec for the Windows executable: a folder with SOMTrack.exe.
#
# Build it with packaging\build_exe.bat, which sets up the environment this
# expects (PYTHONPATH pointing at the repository root).
#
#   standard   (build_exe.bat)       no PHATE, SLISEMAP or PyTorch
#   full       (build_exe.bat full)  PHATE and SLISEMAP; PyTorch is downloaded
#              by the app the first time SLISEMAP is wanted, in the build
#              that suits the GPU (somtrack/addons.py)
#
# PyTorch itself is never bundled: with CUDA it is about 3 GB, more than a
# GitHub release file may hold, and most of it is wasted on a machine
# without a suitable NVIDIA card.

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
FULL = os.environ.get("SOMTRACK_FULL") == "1"

hidden = (collect_submodules("somtrack")          # locales are imported by name
          + collect_submodules("pacmap")
          + collect_submodules("umap", filter=lambda m: "parametric" not in m)
          + collect_submodules("pynndescent")
          + ["sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes"])

datas = []
for pkg in ("cmcrameri", "glasbey", "umap", "pynndescent"):
    datas += collect_data_files(pkg)

excludes = ["torch", "torchvision", "torchaudio", "tensorflow", "keras",
            "IPython", "jupyter", "notebook", "tkinter", "_tkinter", "pytest",
            "PyQt5", "PyQt6", "PySide2", "sphinx", "matplotlib.backends._backend_tk",
            "matplotlib.backends.backend_tkagg"]

if FULL:
    def _no_tests(name):
        return not any(p in name for p in (".tests", ".testing", ".test_", ".conftest"))

    # SLISEMAP and PHATE, and what PyTorch imports from outside itself
    for pkg in ("slisemap", "phate", "graphtools", "pygsp", "tasklogger",
                "deprecated", "wrapt", "seaborn", "sympy", "mpmath", "networkx",
                "jinja2", "markupsafe", "fsspec", "filelock", "typing_extensions"):
        hidden += collect_submodules(pkg, filter=_no_tests)
        datas += collect_data_files(pkg)

    # PyTorch is loaded from outside the bundle, so PyInstaller never sees what
    # it imports.  Every standard-library module goes in, or PyTorch fails on
    # the first one SOMTrack itself happened not to use.
    _skip = {"tkinter", "turtle", "turtledemo", "idlelib", "test", "lib2to3",
             "ensurepip", "venv", "pydoc_data", "this", "antigravity", "_tkinter"}
    for name in sorted(sys.stdlib_module_names):
        if name in _skip:
            continue
        try:
            __import__(name)
        except Exception:                      # POSIX-only, or not built here
            continue
        hidden += collect_submodules(
            name, filter=lambda m: _no_tests(m) and not m.startswith(
                ("idlelib", "tkinter", "test.", "lib2to3", "turtledemo")))

a = Analysis(
    [os.path.join(SPECPATH, "somtrack_app.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hidden,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SOMTrack",
    # A console program, not a windowed one.  Under a windowed (pythonw-like)
    # process joblib's worker processes die on start-up, and sys.stdout is
    # None, which breaks libraries that print.  The app hides the console it
    # owns as soon as it starts (somtrack.parallel.hide_console), so a
    # double-click still shows only the window.
    console=True,
    upx=False,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SOMTrack")
