"""Regression coverage for cluster jobs without an exported browser cache."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.browser_setup import configure_browser_cache


class BrowserCacheTests(unittest.TestCase):
    def test_staged_cache_is_discovered_without_export(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            cache = root / '.playwright-browsers'
            cache.mkdir()
            configure_browser_cache(root)
            self.assertEqual(os.environ['PLAYWRIGHT_BROWSERS_PATH'], str(cache.resolve()))

    def test_explicit_cache_and_hermetic_mode_take_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.playwright-browsers').mkdir()
            for value in ('/custom/browser-cache', '0'):
                with self.subTest(value=value), patch.dict(os.environ, {
                    'PLAYWRIGHT_BROWSERS_PATH': value,
                    'POKERL_CHROME': '/custom/chrome',
                }, clear=True):
                    configure_browser_cache(root)
                    self.assertEqual(os.environ['PLAYWRIGHT_BROWSERS_PATH'], value)
                    self.assertEqual(os.environ['POKERL_CHROME'], '/custom/chrome')

    def test_standard_playwright_cache_used_when_nothing_is_staged(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            configure_browser_cache(Path(directory))
            self.assertNotIn('PLAYWRIGHT_BROWSERS_PATH', os.environ)


if __name__ == '__main__':
    unittest.main()
