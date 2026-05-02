"""Scraper subpackage shim — imports from parent package.

The SPEC §1 directory layout places scraper.py here. The actual implementation
lives at fortuna/scraper.py for simpler imports. This shim re-exports everything
so both import paths work.
"""

from fortuna.scraper import GLOScraper, KapookScraper, SanookScraper, fetch_url

__all__ = ["GLOScraper", "KapookScraper", "SanookScraper", "fetch_url"]
