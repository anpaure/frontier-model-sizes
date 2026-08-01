from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from normalize_ooxml_zip import FIXED_ZIP_TIMESTAMP, normalize_ooxml_zip
from normalize_inspect_ndjson import normalize_inspect_ndjson


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
PACKAGES = (
    OUT / "frontier_parameter_prediction_registry_v2.1_2026-07-17.docx",
    OUT / "frontier_parameter_model_crowd_50pct_2026-07-17.xlsx",
)
INSPECTION_LEDGER = (
    OUT / "frontier_parameter_model_crowd_50pct_2026-07-17.xlsx.inspect.ndjson"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OoxmlDeterminismTest(unittest.TestCase):
    def test_production_packages_have_fixed_unique_members(self) -> None:
        for path in PACKAGES:
            with self.subTest(path=path.name), ZipFile(path) as archive:
                members = archive.infolist()
                self.assertEqual(len(members), len({member.filename for member in members}))
                self.assertTrue(members)
                self.assertTrue(
                    all(member.date_time == FIXED_ZIP_TIMESTAMP for member in members)
                )

    def test_normalizer_is_idempotent_and_content_preserving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.docx"
            from zipfile import ZIP_DEFLATED

            with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("word/document.xml", b"<document>stable</document>")
                archive.writestr("[Content_Types].xml", b"<types />")
            with ZipFile(path) as archive:
                before = {name: archive.read(name) for name in archive.namelist()}

            normalize_ooxml_zip(path)
            first_hash = digest(path)
            normalize_ooxml_zip(path)
            self.assertEqual(first_hash, digest(path))
            with ZipFile(path) as archive:
                after = {name: archive.read(name) for name in archive.namelist()}
                self.assertTrue(
                    all(info.date_time == FIXED_ZIP_TIMESTAMP for info in archive.infolist())
                )
            self.assertEqual(before, after)

    def test_xlsx_random_identifiers_are_canonicalized_consistently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xlsx"
            relationship = b"R0123456789abcdef"
            person = b"{12345678-1234-4234-8234-123456789ABC}"
            with ZipFile(path, "w") as archive:
                archive.writestr(
                    "xl/_rels/workbook.xml.rels",
                    b'<Relationship Id="' + relationship + b'" Target="/xl/worksheets/sheet1.xml" />',
                )
                archive.writestr(
                    "xl/workbook.xml",
                    b'<sheet r:id="' + relationship + b'" />',
                )
                archive.writestr(
                    "xl/persons/person.xml",
                    b'<person id="' + person + b'" />',
                )
                archive.writestr(
                    "xl/threadedcomments/threadedcomment1.xml",
                    b'<threadedComment dT="2026-07-18T12:34:56.789" personId="'
                    + person
                    + b'" />',
                )
            normalize_ooxml_zip(path)
            first_hash = digest(path)
            with ZipFile(path) as archive:
                workbook = archive.read("xl/workbook.xml")
                relationships = archive.read("xl/_rels/workbook.xml.rels")
                people = archive.read("xl/persons/person.xml")
                comments = archive.read("xl/threadedcomments/threadedcomment1.xml")
            self.assertIn(b"R0000000000000001", workbook)
            self.assertIn(b"R0000000000000001", relationships)
            self.assertIn(b"{00000000-0000-4000-8000-000000000001}", people)
            self.assertIn(b"{00000000-0000-4000-8000-000000000001}", comments)
            self.assertIn(b'dT="2026-07-18T00:00:00.000"', comments)
            normalize_ooxml_zip(path)
            self.assertEqual(first_hash, digest(path))

    def test_inspection_runtime_ids_are_canonical_and_data_values_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xlsx.inspect.ndjson"
            thread = "random3"
            author = "{ABCDEF12-3456-4789-8123-ABCDEF123456}"
            source_value = "{AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE}"
            records = [
                {"kind": "workbook", "id": "wb/random1", "sheets": 1},
                {"kind": "sheet", "id": "ws/random2", "name": "Data"},
                {
                    "kind": "thread",
                    "id": f"th/{thread}",
                    "comments": [
                        {
                            "id": thread,
                            "authorId": author,
                            "createdAt": "2026-07-18T12:34:56.789Z",
                        }
                    ],
                    "values": [[source_value]],
                },
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            normalize_inspect_ndjson(path)
            first_hash = digest(path)
            normalized = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(normalized[0]["id"], "wb/000001")
            self.assertEqual(normalized[1]["id"], "ws/000001")
            self.assertEqual(
                normalized[2]["id"],
                "th/000001",
            )
            self.assertEqual(normalized[2]["comments"][0]["id"], "000001")
            self.assertEqual(
                normalized[2]["comments"][0]["authorId"],
                "{00000000-0000-4000-8000-000000000001}",
            )
            self.assertEqual(
                normalized[2]["comments"][0]["createdAt"],
                "2026-07-18T00:00:00.000Z",
            )
            self.assertEqual(normalized[2]["values"][0][0], source_value)
            normalize_inspect_ndjson(path)
            self.assertEqual(first_hash, digest(path))

    def test_production_inspection_ledger_has_only_canonical_runtime_metadata(self) -> None:
        runtime_id = re.compile(r"^(?:wb|ws|ch|sh|img|tbl|th)/\d{6}$")
        bare_id = re.compile(r"^\d{6}$")
        author_id = re.compile(
            r"^\{00000000-0000-4000-8000-\d{12}\}$"
        )
        records = [
            json.loads(line)
            for line in INSPECTION_LEDGER.read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(records)
        seen_runtime_ids: set[str] = set()
        for record in records:
            if "id" in record:
                self.assertRegex(record["id"], runtime_id)
                self.assertNotIn(record["id"], seen_runtime_ids)
                seen_runtime_ids.add(record["id"])
            for comment in record.get("comments", []):
                self.assertRegex(comment["id"], bare_id)
                self.assertRegex(comment["authorId"], author_id)
                self.assertEqual(
                    comment["createdAt"],
                    "2026-07-18T00:00:00.000Z",
                )


if __name__ == "__main__":
    unittest.main()
