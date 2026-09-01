#!/usr/bin/env python3
"""Retrieve and validate the small DICOM sample produced by the Kaggle kernel.

The regular ``kaggle kernels output`` command can create an empty file when a
large kernel output is streamed through a proxy. This downloader obtains the
signed output URL from Kaggle, streams it to a temporary file, validates the
ZIP, and extracts it only after validation succeeds.

The default destination is ``data/local_dicoms_sample3``. Existing non-empty
destinations are treated as already retrieved; pass ``--force`` only when a
fresh copy is explicitly desired.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KERNEL = "mananjain39/rsna-knee-dicom-sample-extract"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/local_dicoms_sample3"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _output_url(kernel: str) -> tuple[str, int | None]:
    try:
        from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi
    except ImportError as exc:
        raise RuntimeError("Install Kaggle tooling first: pip install kaggle kagglesdk") from exc

    api = KaggleApi()
    api.authenticate()
    owner, slug, _ = api.parse_kernel_string(kernel)
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.page_size = 100
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    for item in response.files or []:
        if item.file_name == "dicom_payload.zip":
            size = int(item.size) if getattr(item, "size", None) else None
            return item.url, size
    names = [item.file_name for item in response.files or []]
    raise FileNotFoundError(f"Kernel output has no dicom_payload.zip; files={names[:8]}")


def _manifest(output_dir: Path) -> dict | None:
    path = output_dir / "dicom_payload" / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    args = _args()
    if not os.environ.get("KAGGLE_API_TOKEN") and not (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        raise RuntimeError("Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME/KAGGLE_KEY) before retrieval")

    existing = _manifest(args.output_dir)
    if existing and not args.force:
        print(
            f"Sample already present: {existing.get('n_studies')} studies at {args.output_dir}. "
            "Use --force to refresh."
        )
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")

    url, expected_size = _output_url(args.kernel)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rsna-knee-download-") as temp_dir:
        archive = Path(temp_dir) / "dicom_payload.zip"
        print(f"Downloading {args.kernel}/dicom_payload.zip")
        with requests.get(url, stream=True, timeout=180) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        actual_size = archive.stat().st_size
        if actual_size == 0:
            raise RuntimeError("Kaggle returned a zero-byte archive")
        if expected_size and actual_size != expected_size:
            raise RuntimeError(f"Archive size mismatch: expected {expected_size}, got {actual_size}")
        with zipfile.ZipFile(archive) as payload:
            bad_member = payload.testzip()
            if bad_member:
                raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
            members = payload.namelist()
            if "dicom_payload/manifest.json" not in members:
                raise RuntimeError("Archive is missing dicom_payload/manifest.json")
            print(f"Validated ZIP: {actual_size / 1024 / 1024:.1f} MiB, {len(members)} files")
            if args.archive_only:
                archive_target = args.output_dir.with_suffix(".zip")
                shutil.copy2(archive, archive_target)
                print(f"Saved archive: {archive_target}")
                return
            if args.force and args.output_dir.exists():
                shutil.rmtree(args.output_dir)
            payload.extractall(args.output_dir)

    manifest = _manifest(args.output_dir)
    if not manifest:
        raise RuntimeError("Extraction finished but manifest could not be read")
    print(f"Retrieved {manifest.get('n_studies')} studies at {args.output_dir}")


if __name__ == "__main__":
    main()
