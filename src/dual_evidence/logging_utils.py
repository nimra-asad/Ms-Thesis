"""Shared logging setup."""
from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    """Configure console and rotating-to-file style append logging."""
    destination = Path(log_dir)
    destination.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(destination / "pipeline.log")],
        force=True,
    )
