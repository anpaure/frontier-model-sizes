from __future__ import annotations

import unittest

from vendor_external_inputs import INPUTS, sha256


class VendoredInputTest(unittest.TestCase):
    def test_every_original_input_is_present_and_hash_exact(self) -> None:
        self.assertEqual(len(INPUTS), 7)
        for destination, (_, expected) in INPUTS.items():
            self.assertTrue(destination.is_file(), destination)
            self.assertEqual(sha256(destination), expected, destination)


if __name__ == "__main__":
    unittest.main()
