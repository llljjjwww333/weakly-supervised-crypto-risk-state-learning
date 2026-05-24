from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _insert_front(path: str | Path) -> None:
    text = str(path)
    if text and text not in sys.path:
        sys.path.insert(0, text)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: windows_python_runner.py <script-or-module> [args...]")

    target = sys.argv[1]
    target_args = sys.argv[2:]

    repo_root = Path(__file__).resolve().parents[1]
    user_site = Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages"

    base_site_entries = [
        entry
        for entry in sys.path
        if "SecurityIssueAnalysis\\python\\lib\\site-packages" in entry
        or "SecurityIssueAnalysis\\python\\Lib\\site-packages" in entry
        or "SecurityIssueAnalysis\\python\\site-packages.zip" in entry
    ]
    sys.path = [entry for entry in sys.path if entry not in base_site_entries]
    _insert_front(user_site)
    _insert_front(repo_root)
    sys.path.extend(base_site_entries)

    sys.argv = [target, *target_args]
    if target.endswith(".py"):
        runpy.run_path(str((repo_root / target).resolve()), run_name="__main__")
    else:
        runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
