import unittest
import cv2
import numpy as np
from pixel_whitening import blue_surround, inspect


class SurroundTests(unittest.TestCase):
    def fixture(self):
        mask = np.zeros((100,100), bool)
        mask[48:53,48:53] = True
        hsv = np.full((100,100,3), (110,200,100), np.uint8)
        hsv[mask] = (0,0,240)
        return mask, hsv

    def test_interior_and_neutral_tolerance(self):
        mask, hsv = self.fixture()
        self.assertEqual(blue_surround(mask,hsv,.1)['blue_fraction'], 1)
        hsv[::5] = (0,0,200)
        r = blue_surround(mask,hsv,.1)
        self.assertGreater(r['blue_fraction'], .75)
        self.assertLess(r['blue_fraction'], .85)
        self.assertGreater(r['minimum_sector_blue_fraction'], .35)

    def test_open_background_side_is_not_enclosed(self):
        mask, hsv = self.fixture()
        hsv[:48] = (0,0,30)
        r = blue_surround(mask,hsv,.1)
        self.assertEqual(r['minimum_sector_blue_fraction'], 0)

    def test_truncated_ring_and_scale(self):
        mask,hsv = self.fixture()
        mask[:] = False
        mask[0:4,48:53] = True
        self.assertFalse(blue_surround(mask,hsv,.1)['complete_ring'])
        mask,hsv = self.fixture()
        a=blue_surround(mask,hsv,.1)
        b=blue_surround(np.repeat(np.repeat(mask,2,0),2,1),
                        cv2.resize(hsv,(200,200),interpolation=cv2.INTER_NEAREST),.1)
        self.assertEqual(b['radius_pixels'], a['radius_pixels']*2)
        self.assertEqual(a['blue_fraction'], b['blue_fraction'])

    def test_filter_integration_and_large_exception(self):
        sheet=np.full((1050,1010,3),(140,45,20),np.uint8)
        cv2.rectangle(sheet,(200,220),(203,223),(240,240,240),-1)
        cv2.rectangle(sheet,(300,320),(313,333),(240,240,240),-1)
        result,_=inspect(sheet,.8,.02)
        self.assertEqual(len(result['excluded_candidates']),1)
        self.assertEqual(result['candidate_count'],1)
        self.assertEqual(result['candidates'][0]['retained_by'],'large_mark_exception')
        self.assertEqual(result['candidates'][0]['reason_code'],'L')
        self.assertEqual(result['excluded_candidates'][0]['reason_code'],'R')
        triangle,_=inspect(sheet,.6,.02,'triangle')
        self.assertEqual(triangle['excluded_candidates'][0]['reason_code'],'T')
        self.assertTrue(triangle['excluded_candidates'][0]['display_label'].endswith('T'))

if __name__ == '__main__':
    unittest.main()
