"""
tools/__init__.py
-----------------
Exposes the public search helper so callers can do:

    from tools import run_search
"""

from .search import run_search  # noqa: F401

__all__ = ["run_search"]
