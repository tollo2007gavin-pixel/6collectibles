import unittest
from unittest.mock import patch

import cv2
import numpy as np

import back_landmarks as detector
from whitening_geometry import detect_whitening_back


class PrintedBackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Artificial repeatable print with a plain border and known physical corners.
        cls.body = np.full((504, 360, 3), (110, 55, 35), np.uint8)
        rng = np.random.default_rng(24)
        texture = rng.integers(0, 255, (420, 286), np.uint8)
        cls.body[42:-42, 37:-37] = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        gray = cv2.createCLAHE(2., (8, 8)).apply(cv2.cvtColor(cls.body, cv2.COLOR_BGR2GRAY))
        keys, features = cv2.SIFT_create(nfeatures=650).detectAndCompute(gray, None)
        cls.reference = [("synthetic", np.float32([k.pt for k in keys]), features)]
        cls.expected = np.float32([[116, 49], [434, 82], [413, 511], [75, 472]])
        matrix = cv2.getPerspectiveTransform(detector.FRAME, cls.expected)
        cls.image = cv2.warpPerspective(cls.body, matrix, (530, 570), borderValue=(175, 165, 155))

    def test_included_designs_have_matching_points_and_descriptors(self):
        for _, points, features in detector.reference_features():
            self.assertEqual(len(points), len(features))
            self.assertGreater(len(points), 100)
            self.assertEqual(features.shape[1], 128)

    def test_front_tag_can_be_overturned_by_print_and_physical_edges(self):
        with patch.object(detector, "reference_features", return_value=self.reference):
            quad, info = detect_whitening_back(self.image, "front")
        self.assertIsNotNone(quad, info)
        self.assertLess(np.linalg.norm(quad-self.expected, axis=1).max(), 5)
        self.assertEqual(info["recovery"], "printed_back_landmarks")

    def test_wrong_inner_outline_replaced_even_if_border_check_accepts_it(self):
        center = self.expected.mean(axis=0)
        inner = center + (self.expected-center)*.90
        legacy = dict(recovery="supported_back_contour", sides=[{}]*4)
        with patch.object(detector, "reference_features", return_value=self.reference), \
             patch("whitening_geometry.detect", return_value=(inner, legacy)):
            quad, info = detect_whitening_back(self.image, "back")
        self.assertLess(np.linalg.norm(quad-self.expected, axis=1).max(), 5)
        self.assertEqual(info["previous_reason"], "outline_disagrees_with_print_location")

    def test_partial_card_cannot_generate_four_corners(self):
        with patch.object(detector, "reference_features", return_value=self.reference):
            quad, info = detector.detect_printed_back(self.image[:-150])
        self.assertEqual(info.get("design"), "synthetic")
        self.assertIsNone(quad)

    def test_logo_patch_alone_is_insufficient(self):
        patch_image = self.image.copy()
        patch_image[:230] = patch_image[320:] = patch_image[0, 0]
        with patch.object(detector, "reference_features", return_value=self.reference):
            self.assertIsNone(detector.match_artwork(patch_image))

    def test_no_contrast_at_card_edge_requires_review(self):
        image = np.full((570, 530, 3), self.body[0, 0], np.uint8)
        matrix = cv2.getPerspectiveTransform(detector.FRAME, self.expected)
        cv2.warpPerspective(self.body, matrix, (530, 570), dst=image,
                            borderMode=cv2.BORDER_TRANSPARENT)
        with patch.object(detector, "reference_features", return_value=self.reference):
            quad, info = detector.detect_printed_back(image)
        self.assertIsNone(quad)
        self.assertEqual(info.get("reason"), "back_edge_unconfirmed")


if __name__ == "__main__":
    unittest.main()
