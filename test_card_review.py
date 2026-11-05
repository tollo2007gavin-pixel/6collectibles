import json
from pathlib import Path
import tempfile
import unittest

from card_review import build_batch, local_asset, write_review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name).resolve()
        self.asset = self.out / '_audit/backs/test/original.png'
        self.asset.parent.mkdir(parents=True)
        self.asset.write_bytes(b'example-image-bytes')
        self.row = {'item_id': 'v1|123|0', 'title': '</script><script>alert(1)</script>', 'url': 'javascript:alert(1)',
                    'status': 'reject', 'accepted': False, 'whitening': {'decision': 'not_clean'},
                    'back_packet': {'context': '/home/runner/outputs/run/_audit/backs/test/original.png'}}
        self.state = {'started_utc': '2026-11-04', 'state': 'listing_limit_reached', 'settings': {'pipeline': 'api'}, 'records': [self.row]}

    def test_ci_paths_relocate_and_images_embed(self):
        b = build_batch(self.out, self.state)
        self.assertTrue(b['cards'][0]['images'][0]['src'].startswith('data:image/png;base64,'))
        self.assertEqual(b['cards'][0]['url'], '')
        self.assertNotIn('human_review', b['cards'][0])
        self.assertEqual(b['batch_sha256'], build_batch(self.out, self.state, embed=False)['batch_sha256'])

    def test_traversal_cannot_read_outside_run(self):
        self.assertIsNone(local_asset(self.out, '../../secret'))
        self.assertIsNone(local_asset(self.out, '/tmp/unrelated'))

    def test_originals_from_manifest_included_when_preparation_failed(self):
        m = self.asset.parent / 'manifest.json'
        m.write_text(json.dumps({'photos': [{'original_file': 'original.png'}]}))
        self.row['back_packet'] = {'manifest': str(m)}
        self.assertEqual(len(build_batch(self.out, self.state)['cards'][0]['images']), 1)

    def test_production_filter_and_duplicate_id_validation(self):
        self.assertEqual(build_batch(self.out, self.state, production=True)['cards'], [])
        self.state['records'].append(self.row.copy())
        with self.assertRaises(ValueError):
            build_batch(self.out, self.state)

    def test_safe_portable_html_and_no_network_dependencies(self):
        write_review(self.out, self.state)
        page = (self.out / 'review.html').read_text(encoding='utf-8')
        self.assertNotIn('</script><script>alert', page)
        self.assertNotIn('javascript:alert', page)
        self.assertNotIn('__BATCH_DATA__', page)
        self.assertNotIn('<script src=', page)
        self.assertIn('Save reviewed HTML', page)
        self.assertIn('Export review JSON', page)


if __name__ == '__main__':
    unittest.main()
