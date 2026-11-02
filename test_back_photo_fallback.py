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
        self.assertTrue(p['ready'])
        self.assertEqual(p['input_mode'], 'original_photos')
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

    def test_fallback_request_has_only_originals_and_explicit_visibility_schema(self):
        p = self.prepare({})
        body = luna.request(p['context'], None, None, [])
        content = body['messages'][1]['content']
        self.assertEqual(sum(c['type'] == 'image_url' for c in content), 1)
        self.assertIn('back_visible', body['response_format']['json_schema']['schema']['required'])
        self.assertIn('Never infer clean from missing evidence', body['messages'][0]['content'])
        result = luna.classify(p['context'], None, None, [], lambda body: {'decision': 'needs_review', 'location': '', 'evidence': 'No back visible', 'cost_usd': 0})
        self.assertEqual(result['decision'], 'needs_review')
        self.assertEqual(result['provenance']['input_mode'], 'original_photos')
        self.assertEqual(result['timing']['inspection_mode'], 'parallel')
        self.assertEqual([a['inspection'] for a in result['timing']['attempts']], ['original_1', 'original_2'])

    def test_empty_originals_fail_before_paid_calls(self):
        caller = Mock()
        result = luna.classify([], None, None, [], caller)
        caller.assert_not_called()
        self.assertEqual(result['decision'], 'needs_review')

    def test_visibility_gates_cannot_pass_an_unassessable_back(self):
        p = self.prepare({})
        body = luna.request(p['context'], None, None, [])
        for back, perimeter, decision, expected in [(False, True, 'clean', 'needs_review'), (True, False, 'clean', 'needs_review'), (False, False, 'not_clean', 'needs_review'), (True, False, 'not_clean', 'not_clean'), (True, True, 'clean', 'clean')]:
            answer = dict(decision=decision, location='corner', evidence='test', back_visible=back, perimeter_assessable=perimeter)
            response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'}), patch.object(luna.providers, '_post', return_value=response):
                self.assertEqual(luna.call(body)['decision'], expected)

    def test_missing_visibility_is_invalid(self):
        p = self.prepare({})
        body = luna.request(p['context'], None, None, [])
        answer = dict(decision='clean', location='', evidence='')
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-only'}), patch.object(luna.providers, '_post', return_value=response):
            self.assertTrue(luna.call(body)['invalid'])

    def test_fallback_runs_through_coordinator_and_exports_photos(self):
        out = self.root / 'run'
        audit = out / '_audit'
        audit.mkdir(parents=True)
        settings = runner.Settings(pipeline='api')
        state = runner.new_state(settings)
        state['records'].append(self.row)
        classifier = Mock(return_value={'decision': 'needs_review', 'stages': [], 'cost_usd': 0})
        with patch.object(luna, 'references', return_value=[]):
            runner.finish_listing(self.row, {}, self.client, audit, settings, [], state, lambda: None, classifier=classifier)
        classifier.assert_called_once()
        self.assertIsInstance(classifier.call_args.args[0], list)
        self.assertFalse(self.row['accepted'])
        runner.checkpoint(out, state, settings)
        self.assertIn('Original listing photo 1', (out / 'review.html').read_text())
        self.assertIn('original_photos', (out / 'results.csv').read_text())


if __name__ == '__main__':
    unittest.main()
