import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import card_runner as runner
import luna_whitening as luna


class BackPhotoFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = cv2.imencode('.png', np.zeros((200, 150, 3), np.uint8))[1].tobytes()
        self.client = Mock()
        self.client.download.return_value = self.raw
        self.row = {'item_id': 'v1|123|0', 'image_url': 'primary', 'card_id': '22',
                    'border_type': 'yellow', 'title': 'Test', 'url': ''}

    def prepare(self, detail, **kwargs):
        return runner.prepare_back(self.client, detail, self.row, self.root, runner.Settings(**kwargs))

    def test_primary_only_is_not_discarded_and_unknown_is_not_called_absent(self):
        p = self.prepare({})
        self.assertFalse(p['ready'])
        self.assertEqual(p['input_mode'], 'sheet_preparation_failed')
        self.assertEqual(len(p['context']), 1)
        self.assertIsNone(p['sheet'])
        self.assertIsNone(p['tips'])
        self.assertEqual(Path(p['context'][0]).read_bytes(), self.raw)
        self.client.download.assert_called_once_with('primary')

    def test_duplicate_urls_do_not_consume_extra_photo_budget(self):
        p = self.prepare({'additionalImages': [{'imageUrl': u} for u in ['a', 'a', 'b', 'c', 'primary']]}, max_back_photos=2)
        self.assertEqual([c.args[0] for c in self.client.download.call_args_list], ['a', 'b', 'primary'])
        self.assertEqual(len(p['context']), 3)

    def test_no_readable_images_does_not_trigger_model(self):
        self.client.download.side_effect = ValueError('download failed')
        p = self.prepare({})
        self.assertFalse(p['ready'])
        self.assertEqual(p['reason'], 'no_readable_listing_photos')


    def test_classifier_rejects_original_only_before_api(self):
        caller = Mock()
        result = luna.classify(["original.jpg"], None, None, [], caller)
        caller.assert_not_called()
        self.assertEqual(result['decision'], 'needs_review')
        self.assertIn('four-corner', result['error'])

    def test_failed_sheets_remain_visible_without_paid_assessment(self):
        out = self.root / 'run'
        audit = out / '_audit'
        audit.mkdir(parents=True)
        settings = runner.Settings(pipeline='api')
        state = runner.new_state(settings)
        state['records'].append(self.row)
        classifier = Mock()
        runner.finish_listing(self.row, {}, self.client, audit, settings, [], state, lambda: None, classifier=classifier)
        classifier.assert_not_called()
        self.assertFalse(self.row['accepted'])
        self.assertEqual(self.row['reason'], 'back_corner_sheet_unavailable')
        runner.checkpoint(out, state, settings)
        self.assertIn('Original listing photo 1', (out / 'review.html').read_text())

    def test_prepared_card_sends_both_generated_sheets(self):
        pixels = np.full((900, 900, 3), 205, np.uint8)
        cv2.rectangle(pixels, (220,160), (620,720), (150,30,15), -1)
        cv2.rectangle(pixels, (250,190), (590,690), (190,140,90), -1)
        self.client.download.return_value = cv2.imencode('.png',pixels)[1].tobytes()
        packet = self.prepare({})
        self.assertTrue(packet['ready'])
        self.assertEqual(packet['input_mode'], 'validated_crops')
        for name in ['context','sheet','tips']:
            self.assertIsNotNone(cv2.imread(packet[name]))
        bodies=[]
        def caller(body):
            bodies.append(body)
            return {'decision':'clean','location':'','evidence':'test only','cost_usd':0}
        result=luna.classify(packet['context'],packet['sheet'],packet['tips'],[],caller,parallel=False)
        self.assertEqual(result['decision'],'clean')
        self.assertEqual([sum(x['type']=='image_url' for x in b['messages'][1]['content']) for b in bodies],[2,3])
        self.assertEqual(bodies[0]['messages'][1]['content'][-1],luna.image(packet['sheet']))
        self.assertEqual(bodies[1]['messages'][1]['content'][-1],luna.image(packet['tips']))


if __name__ == '__main__':
    unittest.main()
