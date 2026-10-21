"""Locate browsers staged with the project before starting Playwright."""
import os
from pathlib import Path


def configure_browser_cache(root: Path) -> None:
    """Prefer a staged cache when unset; retain explicit and standard defaults."""
    cache = root / '.playwright-browsers'
    if 'PLAYWRIGHT_BROWSERS_PATH' not in os.environ and cache.is_dir():
        # An absolute path also works in spawned workers and outside the repo cwd.
        os.environ['PLAYWRIGHT_BROWSERS_PATH'] = str(cache.resolve())
