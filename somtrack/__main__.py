"""``python -m somtrack`` -- launch the desktop app, or the CLI with arguments."""

from __future__ import annotations

import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-"):
        from .cli import main as cli_main

        return cli_main(argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        print("  python -m somtrack              launch the GUI")
        print("  python -m somtrack run ...      batch analysis, see 'run --help'")
        print("  python -m somtrack demo ...     write demo data")
        return 0

    from .ui.app import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
