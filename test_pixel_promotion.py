import json
import tempfile
import unittest
from pathlib import Path

from edge_classifier import DEFAULTS
from pixel_active_settings import ACTIVE_PATH, active_thresholds
from pixel_training import CONFIG_SCHEMA, EXPOSURE_SCHEMA, code_fingerprint, digest
from promote_pixel_settings import promote


class PixelPromotionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "review_labels").mkdir()
        self.artifact = self.root / "downloads"
        self.artifact.mkdir()
        self.selected = dict(DEFAULTS, noise_multiple=4.8)
        self.active = self.root / ACTIVE_PATH.name
        config = dict(schema=CONFIG_SCHEMA, code_fingerprint=code_fingerprint(), core_thresholds=self.selected)
        config["config_sha256"] = digest(config)
        self.report = dict(mode="improve", decision="recommend_update", active_thresholds=dict(DEFAULTS),
                           config_sha256=config["config_sha256"])
        exposure = dict(schema=EXPOSURE_SCHEMA, config_sha256=config["config_sha256"], exposed_aliases=["group-1"])
        for file, value in [("report.json", self.report), ("recommended-settings.json", self.selected),
                            ("frozen-config.json", config), ("validation-exposure.json", exposure)]:
            (self.artifact / file).write_text(json.dumps(value))
        (self.root / "review_labels" / "validation-exposure-123.json").write_text(json.dumps(exposure))

    def test_apply_and_repeat(self):
        self.assertEqual(active_thresholds(self.active), DEFAULTS)
        promote(self.artifact, "123", self.root)
        self.assertEqual(active_thresholds(self.active), self.selected)
        original = self.active.read_bytes()
        promote(self.artifact, "123", self.root)
        self.assertEqual(self.active.read_bytes(), original)

    def test_refuse_stale_and_unrecommended_comparisons(self):
        self.active.write_text(json.dumps(dict(core_thresholds=dict(DEFAULTS, min_area=48))))
        with self.assertRaisesRegex(ValueError, "live settings"):
            promote(self.artifact, "123", self.root)
        self.report["decision"] = "keep_current"
        (self.artifact / "report.json").write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "did not recommend"):
            promote(self.artifact, "123", self.root)

    def test_require_selection_matches_frozen_config(self):
        (self.artifact / "recommended-settings.json").write_text(json.dumps(dict(DEFAULTS, min_area=32)))
        with self.assertRaisesRegex(ValueError, "differ"):
            promote(self.artifact, "123", self.root)
        self.assertFalse(self.active.exists())
