"""Behavior tests for conservative corner-background rejection."""
import unittest

import cv2
import numpy as np

from edge_uncertainty import estimate, classify


class EdgeUncertaintyTests(unittest.TestCase):
    def corner(self, size=500, shift=0):
        # TR corner: a rounded/chamfered blue card continues into left/bottom.
        crop = np.full((size, size, 3), 35, np.uint8)
        points = np.array([[0, 100], [220, 100], [300, 150], [350, 220],
                           [350, 500], [0, 500]], np.float32)
        points[:4, 1] += shift
        points = np.rint(points * (size / 500)).astype(np.int32)
        cv2.fillConvexPoly(crop, points, (140, 45, 20))
        return crop

    def component(self, size, box):
        mask = np.zeros((size, size), bool)
        x0,y0,x1,y1 = [round(v*size/500) for v in box]
        mask[y0:y1, x0:x1] = True
        return mask

    def test_detached_exterior_spot_removed_and_edge_chip_retained(self):
        crop = self.corner()
        info, background, zone = estimate(crop, 'TR')
        self.assertEqual(info['status'], 'agreement')
        dust = self.component(500, (420, 40, 428, 48))
        edge_chip = self.component(500, (90, 95, 98, 103))
        self.assertTrue(classify(dust, background, zone)['exclude'])
        self.assertFalse(classify(edge_chip, background, zone)['exclude'])

    def test_raw_card_support_distinguishes_margin_without_rejecting_it(self):
        details={}
        _,background,zone=estimate(self.corner(),'TR',details=details)
        near=self.component(500,(90,85,98,90))
        chip=self.component(500,(90,98,98,106))
        self.assertFalse(details['possible_interior'][near].any())
        self.assertTrue(details['possible_interior'][chip].any())
        self.assertFalse(classify(near,background,zone)['exclude'])
        self.assertGreater(details['outside_distance'][near].min(),0)

    def test_a_protrusion_touching_uncertainty_is_not_rejected_by_its_centroid(self):
        _, background, zone = estimate(self.corner(), 'TR')
        protrusion = self.component(500, (90, 15, 96, 94))
        result = classify(protrusion, background, zone)
        self.assertGreater(result['background_fraction'], .7)
        self.assertGreater(result['uncertainty_fraction'], 0)
        self.assertFalse(result['exclude'])

    def test_edge_location_follows_the_image_and_scales(self):
        for size in (250, 500, 1000):
            with self.subTest(size=size):
                first, background, zone = estimate(self.corner(size), 'TR')
                moved, shifted_background, shifted_zone = estimate(self.corner(size, 50), 'TR')
                self.assertEqual(first['status'], 'agreement')
                self.assertEqual(moved['status'], 'agreement')
                spot = self.component(size, (100, 95, 108, 103))
                self.assertFalse(classify(spot, background, zone)['exclude'])
                self.assertTrue(classify(spot, shifted_background, shifted_zone)['exclude'])

    def test_all_corner_orientations(self):
        transforms = {'TR': lambda x: x, 'TL': np.fliplr,
                      'BR': np.flipud, 'BL': lambda x: np.flipud(np.fliplr(x))}
        dust = self.component(500, (420, 40, 428, 48))
        chip = self.component(500, (90, 95, 98, 103))
        for name, transform in transforms.items():
            with self.subTest(corner=name):
                info, background, zone = estimate(transform(self.corner()).copy(), name)
                self.assertEqual(info['status'], 'agreement')
                self.assertTrue(classify(transform(dust), background, zone)['exclude'])
                self.assertFalse(classify(transform(chip), background, zone)['exclude'])

    def test_no_blue_or_detached_blue_object_fails_open(self):
        gray = np.full((500, 500, 3), 170, np.uint8)
        detached = gray.copy()
        cv2.circle(detached, (250, 250), 120, (140, 45, 20), -1)
        for crop in (gray, np.zeros_like(gray), detached):
            info, background, zone = estimate(crop, 'TR')
            self.assertEqual(info['status'], 'unconfirmed')
            self.assertFalse(background.any())
            self.assertFalse(classify(self.component(500, (30, 30, 40, 40)), background, zone)['exclude'])

    def test_blue_background_cannot_support_aggressive_rejection(self):
        blue_background = np.full((500, 500, 3), (140, 45, 20), np.uint8)
        _, background, zone = estimate(blue_background, 'TR')
        self.assertFalse(background.any())
        self.assertFalse(classify(self.component(500, (30, 30, 40, 40)), background, zone)['exclude'])

    def test_glare_connected_to_edge_is_not_confident_background(self):
        crop=self.corner()
        cv2.rectangle(crop,(70,40),(110,130),(220,220,220),-1)
        info,background,zone=estimate(crop,'TR')
        self.assertGreater(info['bright_connected_fraction'],0)
        self.assertFalse(classify(self.component(500,(85,45,92,55)),background,zone)['exclude'])


if __name__ == '__main__':
    unittest.main()
