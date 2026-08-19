import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from validate_4m import (
    format_masking_summary,
    masking_summary,
    write_masking_reports,
)


class MaskReportingTest(unittest.TestCase):

    @staticmethod
    def _mask(masked, total):
        return torch.cat((torch.ones(masked), torch.zeros(total - masked)))

    def test_reports_each_modality_and_actual_global_percentage(self):
        summary = masking_summary(
            self._mask(192, 196),
            self._mask(140, 196),
            self._mask(121, 196),
            self._mask(7, 25),
        )

        self.assertEqual(summary['rgb']['masked_tokens'], 192)
        self.assertEqual(summary['rgb']['visible_tokens'], 4)
        self.assertAlmostEqual(summary['rgb']['masked_percent'], 97.9591837)
        self.assertEqual(summary['parameters']['total_tokens'], 25)
        self.assertAlmostEqual(summary['parameters']['masked_percent'], 28.0)
        self.assertEqual(summary['overall']['masked_tokens'], 460)
        self.assertEqual(summary['overall']['total_tokens'], 613)
        self.assertAlmostEqual(summary['overall']['masked_percent'], 100 * 460 / 613)

        report = format_masking_summary('Sorghum_10016_01', summary)
        self.assertIn('RGB 192/196 (97.96%)', report)
        self.assertIn('Overall 460/613 (75.04%)', report)

    def test_empty_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one token'):
            masking_summary(
                torch.empty(0), torch.ones(1), torch.ones(1), torch.ones(1))

    def test_writes_one_json_and_csv_row_per_sample(self):
        summary = masking_summary(
            self._mask(170, 196),
            self._mask(140, 196),
            self._mask(137, 196),
            self._mask(13, 25),
        )
        records = [{'sample': 'Sorghum_10007_00', 'masking': summary}]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_masking_reports(out_dir, records)
            saved_json = json.loads(
                (out_dir / 'masking_per_sample.json').read_text())
            with open(out_dir / 'masking_per_sample.csv', newline='') as f:
                saved_csv = list(csv.DictReader(f))

        self.assertEqual(saved_json, records)
        self.assertEqual(saved_csv[0]['sample'], 'Sorghum_10007_00')
        self.assertEqual(saved_csv[0]['point_cloud_masked_tokens'], '137')
        self.assertAlmostEqual(
            float(saved_csv[0]['rgb_masked_percent']), 100 * 170 / 196)


if __name__ == '__main__':
    unittest.main()
