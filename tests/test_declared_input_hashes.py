from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audit_declared_input_hashes import audit_artifacts


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"


class DeclaredInputHashAuditTest(unittest.TestCase):
    def test_recognizes_supported_contract_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources/input.txt"
            source.parent.mkdir()
            source.write_text("immutable input\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            artifact = root / "artifact.json"
            artifact.write_text(
                json.dumps(
                    {
                        "path_map": {"sources/input.txt": digest},
                        "record": {"path": "sources/input.txt", "sha256": digest},
                        "paired": {
                            "input_file": "sources/input.txt",
                            "input_sha256": digest,
                        },
                        "split_maps": {
                            "source_files": {"input": "sources/input.txt"},
                            "source_hashes": {"input": digest},
                            "coverage_source_files": {"negative": "sources/input.txt"},
                            "coverage_source_hashes": {"negative": digest},
                        },
                        "selected_value_contract": {
                            "path": "sources/input.txt",
                            "selector": "example.value",
                            "value": {"example": 1},
                            "sha256": hashlib.sha256(b"selected value").hexdigest(),
                        },
                        "archive": {
                            "required_member_sha256": {
                                "inner.csv": hashlib.sha256(b"zip member").hexdigest()
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            declarations, issues = audit_artifacts([artifact], root=root)
            self.assertGreaterEqual(len(declarations), 1)
            self.assertEqual(issues, [])
            self.assertNotIn(
                "inner.csv", {row.declared_path for row in declarations}
            )

    def test_reports_missing_and_changed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            changed = root / "changed.txt"
            changed.write_text("new bytes", encoding="utf-8")
            old_digest = hashlib.sha256(b"old bytes").hexdigest()
            missing_digest = hashlib.sha256(b"missing bytes").hexdigest()
            artifact = root / "artifact.json"
            artifact.write_text(
                json.dumps(
                    {
                        "changed.txt": old_digest,
                        "missing.txt": missing_digest,
                    }
                ),
                encoding="utf-8",
            )
            _, issues = audit_artifacts([artifact], root=root)
            self.assertEqual(
                {(issue.declared_path, issue.issue) for issue in issues},
                {
                    ("changed.txt", "sha256_mismatch"),
                    ("missing.txt", "missing"),
                },
            )

    def test_expands_home_relative_declared_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            source = home / "portable-input.txt"
            source.write_text("portable input\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            artifact = home / "artifact.json"
            artifact.write_text(
                json.dumps({"~/portable-input.txt": digest}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HOME": str(home)}):
                _, issues = audit_artifacts([artifact], root=home / "elsewhere")
            self.assertEqual(issues, [])

    def test_generated_audits_have_no_stale_declared_inputs(self) -> None:
        _, issues = audit_artifacts(OUT.glob("*.json"), root=ROOT)
        self.assertEqual(
            issues,
            [],
            "Generated audit provenance is stale:\n"
            + "\n".join(
                f"{issue.artifact}{issue.json_pointer}: {issue.declared_path} "
                f"({issue.issue})"
                for issue in issues
            ),
        )


if __name__ == "__main__":
    unittest.main()
