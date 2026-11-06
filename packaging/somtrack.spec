# PyInstaller spec for the Windows executable: a folder with SOMTrack.exe.
#
# Build it with packaging\build_exe.bat, which sets up the environment this
# expects (PYTHONPATH pointing at the repository root).
#
# Left out on purpose: PyTorch and SLISEMAP (several GB with CUDA) and PHATE.
# The app shows those methods as not included in the executable; the Python
# installation (install.bat) has all three.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

hidden = (collect_submodules("somtrack")          # locales are imported by name
          + collect_submodules("pacmap")
          + collect_submodules("umap", filter=lambda m: "parametric" not in m)
          + collect_submodules("pynndescent")
          + ["sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes"])

datas = []
for pkg in ("cmcrameri", "glasbey", "umap", "pynndescent"):
    datas += collect_data_files(pkg)

excludes = ["torch", "tensorflow", "keras", "slisemap", "phate", "graphtools",
            "IPython", "jupyter", "notebook", "tkinter", "_tkinter", "pytest",
            "PyQt5", "PyQt6", "PySide2", "sphinx", "matplotlib.backends._backend_tk",
            "matplotlib.backends.backend_tkagg"]

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
