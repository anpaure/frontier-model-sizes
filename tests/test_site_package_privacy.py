from __future__ import annotations

import json
import re
import tarfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/frontier-parameter-lab-site.tar.gz"
RESPONDENT = re.compile(r"^Respondent R\d{2}$")


class SitePackagePrivacyTest(unittest.TestCase):
    def test_archive_contains_current_anonymous_site_data(self) -> None:
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            names = set(archive.getnames())
            self.assertIn("dist/client/data/forecast-model.json", names)
            payload = json.load(
                archive.extractfile("dist/client/data/forecast-model.json")
            )
            contributors = [
                contributor
                for model in payload["models"]
                for contributor in model["crowd"]["contributors"]
            ]
            self.assertEqual(len(contributors), 40)
            self.assertTrue(all(RESPONDENT.fullmatch(value) for value in contributors))
            self.assertTrue(all(not name.startswith(("/", "../")) for name in names))

    def test_archive_metadata_is_reproducible(self) -> None:
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            for member in archive.getmembers():
                self.assertEqual(member.mtime, 0)
                self.assertEqual(member.uid, 0)
                self.assertEqual(member.gid, 0)
                self.assertEqual(member.uname, "")
                self.assertEqual(member.gname, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
