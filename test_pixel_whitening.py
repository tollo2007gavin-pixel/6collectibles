import unittest
import cv2
import numpy as np
from card_utils import corner_closeup_sheet
from pixel_whitening import corner_tiles, inspect, inspection_band


class CornerInspectionTests(unittest.TestCase):
    def sheet(self):
        card = np.full((2228, 1628, 3), (140, 45, 20), np.uint8)
        return corner_closeup_sheet(card, 'back', tile=100)

    def test_real_sheet_layout_and_caption_exclusion(self):
        sheet = self.sheet()
        tiles = corner_tiles(sheet)
        self.assertEqual([t[0] for t in tiles], ['TL', 'TR', 'BL', 'BR'])
        for _, _, _, crop in tiles:
            self.assertEqual(crop.shape, (100, 100, 3))
            np.testing.assert_array_equal(crop[50, 50], [140, 45, 20])
        result, overlay = inspect(sheet)
        self.assertEqual(result['candidate_count'], 0)
        self.assertEqual(overlay.shape, sheet.shape)

    def test_corner_location_and_source_preservation(self):
        sheet = self.sheet()
        cv2.rectangle(sheet, (150, 60), (162, 72), (240, 240, 240), -1)
        original = sheet.copy()
        result, overlay = inspect(sheet,1.1)
        self.assertGreater(result['candidate_count'], 0)
        self.assertTrue(all(c['corner']=='TR' for c in result['candidates']))
        np.testing.assert_array_equal(sheet, original)
        self.assertEqual(overlay.shape, sheet.shape)
        self.assertEqual(result, inspect(sheet,1.1)[0])

    def test_unrecognized_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            inspect(np.zeros((600, 800, 3), np.uint8))

    def test_large_interior_spots_do_not_bypass_boundary(self):
        sheet = np.zeros((1050, 1010, 3), np.uint8)
        for _, _, _, crop in corner_tiles(sheet):
            crop[:] = (140, 45, 20)
        cv2.rectangle(sheet, (200, 220), (203, 223), (240,240,240), -1)
        cv2.rectangle(sheet, (300, 320), (313, 333), (240,240,240), -1)
        result, _ = inspect(sheet)
        self.assertEqual(len(result['candidates']), 0)

    def test_band_reflects_with_corner_orientation(self):
        crop = np.zeros((500, 500, 3), np.uint8)
        cv2.rectangle(crop, (0, 125), (360, 499), (140,45,20), -1)
        tr, _ = inspection_band('TR', crop)
        tl, _ = inspection_band('TL', np.fliplr(crop).copy())
        # Polygon rasterization can differ by a few boundary pixels on reflection.
        self.assertLess(np.count_nonzero(tl != np.fliplr(tr)), 10)
        shifted = np.zeros_like(crop)
        shifted[30:] = crop[:-30]
        moved, _ = inspection_band('TR', shifted)
        self.assertTrue(tr[125, 150])
        self.assertTrue(tr[20, 150])  # Far outside the card: no outer cutoff.
        self.assertFalse(tr[145, 150])
        self.assertTrue(moved[145, 150])
        self.assertTrue(moved[155, 150])


if __name__ == '__main__':
    unittest.main()
