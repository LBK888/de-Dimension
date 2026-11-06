"""Print the package version from pyproject.toml (used by build_exe.bat)."""

import pathlib
import tomllib

root = pathlib.Path(__file__).resolve().parent.parent
print(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
      ["project"]["version"])
