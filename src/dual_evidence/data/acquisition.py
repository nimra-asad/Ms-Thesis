"""Download the public REWD-P archive and companion files from LUMS."""
from __future__ import annotations

import logging
import zipfile
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

FILES = {
    "rewdp_dataset.zip": "https://web.lums.edu.pk/~eig/gallery/datasets/rewdp/rewdp_dataset.zip",
    "weather_dataset.zip": "https://web.lums.edu.pk/~eig/gallery/datasets/rewdp/weather_dataset.zip",
    "metadata.xlsx": "https://web.lums.edu.pk/~eig/gallery/datasets/rewdp/metadata_ultimate.xlsx",
}


def download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    """Download one public source file atomically, reusing a nonempty local copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Reusing existing file %s (%d bytes)", destination, destination.stat().st_size)
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "dual-evidence-research/0.1"})
    try:
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if offset and response.status != 206:
                offset = 0
            mode = "ab" if offset else "wb"
            with temporary.open(mode) as out:
                if offset and response.status != 206:
                    out.seek(0)
                    out.truncate()
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError(f"Downloaded empty file from {url}")
        temporary.replace(destination)
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        raise
    logger.info("Downloaded %s (%d bytes)", destination, destination.stat().st_size)
    return destination


def acquire_all(destination: str | Path = "data/raw") -> list[Path]:
    """Acquire all official files listed by LUMS Energy Institute."""
    root = Path(destination)
    return [download_file(url, root / name) for name, url in FILES.items()]


def extract_archive(archive: str | Path, destination: str | Path) -> list[Path]:
    """Extract a trusted research archive while rejecting path traversal entries."""
    archive_path = Path(archive)
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as bundle:
        for member in bundle.infolist():
            resolved = (target / member.filename).resolve()
            if resolved != target and target not in resolved.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
        bundle.extractall(target)
        extracted = [target / item.filename for item in bundle.infolist() if not item.is_dir()]
    return extracted
