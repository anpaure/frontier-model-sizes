from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
HOME_PATH = re.compile(r"(?:/|\\)\s*users\s*(?:/|\\)\s*[a-z0-9._-]+", re.IGNORECASE)


def tracked_images() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8")
    return [
        ROOT / value
        for value in output.split("\0")
        if value and Path(value).suffix.lower() in IMAGE_SUFFIXES
    ]


def inspect_image(path: Path, tesseract: str) -> tuple[Path, int, bool, bool]:
    process = subprocess.run(
        [tesseract, str(path), "stdout", "--psm", "6"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    visible_text = process.stdout.decode("utf-8", errors="ignore")
    metadata_parts: list[str] = []
    try:
        with Image.open(path) as image:
            metadata_parts.extend(str(value) for value in image.info.values())
            try:
                metadata_parts.extend(str(value) for value in image.getexif().values())
            except Exception:
                pass
    except Exception:
        return path, process.returncode, bool(HOME_PATH.search(visible_text)), True
    metadata_text = "\n".join(metadata_parts)
    return (
        path,
        process.returncode,
        bool(HOME_PATH.search(visible_text)),
        bool(HOME_PATH.search(metadata_text)),
    )


class ImagePrivacyTest(unittest.TestCase):
    def test_tracked_images_expose_no_local_user_home_paths(self) -> None:
        tesseract = shutil.which("tesseract")
        self.assertIsNotNone(tesseract, "tesseract is required for the image privacy gate")
        paths = tracked_images()
        self.assertGreater(len(paths), 0)
        workers = min(8, max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(lambda path: inspect_image(path, tesseract), paths))
        failures = [str(path.relative_to(ROOT)) for path, code, _, _ in results if code]
        visible_leaks = [
            str(path.relative_to(ROOT)) for path, _, visible, _ in results if visible
        ]
        metadata_leaks = [
            str(path.relative_to(ROOT)) for path, _, _, metadata in results if metadata
        ]
        self.assertEqual(failures, [], f"OCR failed for tracked images: {failures}")
        self.assertEqual(
            visible_leaks,
            [],
            f"tracked images visibly expose local home paths: {visible_leaks}",
        )
        self.assertEqual(
            metadata_leaks,
            [],
            f"tracked image metadata exposes local home paths: {metadata_leaks}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
