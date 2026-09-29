"""Compatibility entry point for hosts that start with ``gunicorn app:app``."""

from backend.main import app

__all__ = ['app']
