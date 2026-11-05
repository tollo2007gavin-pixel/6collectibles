import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import review_corners as rc


class ManualCornersTests(unittest.TestCase):
    def setUp(self):
        image = np.full((600, 450, 3), 190, np.uint8)
        cv2.rectangle(image, (50, 50), (400, 550), (160, 30, 20), -1)
        self.raw = cv2.imencode('.png', image)[1].tobytes()
        digest = hashlib.sha256(self.raw).hexdigest()
        self.correction = dict(image_file='../not-a-local-path.png', image_sha256=digest,
                               width=450, height=600, points=[[.11,.08],[.89,.08],[.89,.92],[.11,.92]])
        self.batch = dict(batch_id='batch', batch_sha256='batch-hash', cards=[dict(id='../../card', title='Card', url='',
            pipeline_decision={'whitening':'clean'}, images=[dict(file=self.correction['image_file'], sha256=digest,
            label='Original listing photo', src='data:image/png;base64,' + base64.b64encode(self.raw).decode())])])
        self.review = dict(batch_id='batch', batch_sha256='batch-hash', cards=[dict(id='../../card',
            human_review=dict(assessment='incorrect', corrected_decision='misaligned', notes='Preserve me', manual_corners=self.correction))])

    def html(self, path):
        path.write_text('<script>throw "Do not execute me"</script><script id="batch-data" type="application/json">'+json.dumps(self.batch)+
                        '</script><script id="saved-review" type="application/json">'+json.dumps(self.review)+'</script>', encoding='utf-8')

    def test_prepare_preserves_photo_and_notes_without_reusing_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp)/'input.html', Path(tmp)/'output'
            self.html(source)
            with patch('luna_whitening.call', side_effect=AssertionError('No paid calls')):
                state = rc.prepare(source, out)
            row = state['records'][0]
            self.assertEqual(row['whitening']['decision'], 'not_run')
            self.assertEqual(row['previous_pipeline_decision']['whitening'], 'clean')
            self.assertEqual(row['source_human_review']['notes'], 'Preserve me')
            self.assertEqual((out/'card_1/original.png').read_bytes(), self.raw)
            for name in ['sheet.jpg','tips.png','outline.jpg']:
                self.assertIsNotNone(cv2.imread(str(out/'card_1'/name)))
            self.assertIn('manual_crops', (out/'review.html').read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError): rc.prepare(source, out)

    def test_mismatched_hash_rejected(self):
        self.correction['image_sha256'] = 'wrong'
        with self.assertRaises(ValueError): rc.selected_images(self.batch, self.review)

    def test_changed_image_bytes_rejected(self):
        self.batch['cards'][0]['images'][0]['src']='data:image/png;base64,YWJj'
        with self.assertRaises(ValueError): rc.selected_images(self.batch, self.review)

    def test_generated_sheet_cannot_be_selected(self):
        self.batch['cards'][0]['images'][0]['label']='Corner close-ups'
        with self.assertRaises(ValueError): rc.selected_images(self.batch, self.review)

    def test_wrong_batch_and_duplicate_cards_rejected(self):
        self.review['batch_id']='other'
        with self.assertRaises(ValueError): rc.selected_images(self.batch, self.review)
        self.review['batch_id']='batch';self.review['cards']*=2
        with self.assertRaises(ValueError): rc.selected_images(self.batch, self.review)

    def test_invalid_corners(self):
        for points in [[[0,0],[1,1],[1,0],[0,1]], [[0,0],[0,1],[1,1],[1,0]],
                       [[0,0],[.01,0],[.01,.01],[0,.01]], [[0,0],[1,0],[1,2],[0,1]],
                       [[0,0],[1,0],[1,float('nan')],[0,1]], [[1,1],[0,1],[0,0],[1,0]]]:
            with self.subTest(points=points), self.assertRaises(ValueError):
                rc.validate_corners({**self.correction,'points':points}, 450,600)
        with self.assertRaises(ValueError): rc.validate_corners(self.correction,451,600)

    def test_explicit_reassessment_uses_new_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp)/'input.html', Path(tmp)/'output'
            self.html(source)
            def classify(context,sheet,tips,refs):
                self.assertEqual(Path(context).read_bytes(), self.raw)
                self.assertTrue(Path(sheet).is_file() and Path(tips).is_file())
                return {'decision':'not_clean'}
            with patch('luna_whitening.references', return_value=['validated-reference']):
                state=rc.prepare(source,out,classifier=classify,bank=['bank'])
            self.assertEqual(state['records'][0]['status'],'reject')
            self.assertEqual(state['state'],'manual_reassessment_complete')


if __name__ == '__main__': unittest.main()
