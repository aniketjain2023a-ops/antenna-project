"""Zenodo antenna dataset, record 15866865 (https://zenodo.org/records/15866865).

Status: NOT INSPECTED. The record could not be reached from the environment in which this
project was built (zenodo.org was blocked by the network policy), so its files, schema, units,
license and antenna type are unknown. ``download`` fetches the record metadata and files through
the public Zenodo REST API and writes a manifest; there is deliberately NO preprocessing pipeline
until the files have been inspected, so the application never offers this dataset unverified.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from datasets.public import DatasetUnavailable, raw_dir
from utils.validation import file_checksum

KEY = "zenodo"
RECORD_ID = "15866865"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
RECORD_URL = f"https://zenodo.org/records/{RECORD_ID}"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "MetaOpt-AI dataset downloader"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return resp.read()


def download(dest: Path | None = None, max_bytes: int = 2_000_000_000) -> dict:
    """Download the record's metadata and files unchanged into data/raw/zenodo."""
    dest = Path(dest or raw_dir(KEY))
    dest.mkdir(parents=True, exist_ok=True)
    try:
        meta = json.loads(_get(API_URL))
    except Exception as exc:  # network policy, DNS, HTTP errors
        raise DatasetUnavailable(f"Could not reach {API_URL}: {exc}") from exc
    (dest / "record_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files = meta.get("files") or []
    if not files:
        raise DatasetUnavailable("The Zenodo record lists no downloadable files.")
    total = sum(int(f.get("size", 0)) for f in files)
    if total > max_bytes:
        raise DatasetUnavailable(f"Record files total {total / 1e9:.1f} GB, above the {max_bytes / 1e9:.0f} GB limit.")
    saved = {}
    for f in files:
        name = Path(f["key"]).name
        (dest / name).write_bytes(_get(f["links"]["self"], timeout=600))
        expected = str(f.get("checksum", ""))
        actual_md5 = None
        if expected.startswith("md5:"):
            import hashlib

            actual_md5 = hashlib.md5((dest / name).read_bytes()).hexdigest()  # noqa: S324 - integrity check only
            if actual_md5 != expected[4:]:
                raise DatasetUnavailable(f"Checksum mismatch for {name}.")
        saved[name] = {"bytes": (dest / name).stat().st_size, "sha256": file_checksum(dest / name),
                       "zenodo_checksum": expected}
    m = meta.get("metadata", {})
    manifest = {
        "source": RECORD_URL, "doi": meta.get("doi"), "title": m.get("title"),
        "license": (m.get("license") or {}).get("id") if isinstance(m.get("license"), dict) else m.get("license"),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": saved,
        "status": "downloaded; schema NOT yet inspected — no preprocessing pipeline exists yet",
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
