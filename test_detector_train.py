import json
import hashlib
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import detector_train as d


class DetectorTests(unittest.TestCase):
    def test_multiple_predictions_cannot_claim_the_same_chip(self):
        box = [10,5,20,9]
        self.assertEqual(d.matched_count([box,box], [box]), 1)
        self.assertEqual(d.matched_count([box], [box,box]), 1)

    def test_paired_scores_and_fully_reviewed_clean_cards(self):
        box = [10,5,20,9]
        rows = [dict(group_id='a', corner=c, boxes=[], baseline_boxes=[]) for c in ['TL','TR','BR','BL']]
        rows.append(dict(group_id='b', corner='TL', boxes=[box], baseline_boxes=[box]))
        pred = [dict(boxes=[box], scores=[s]) for s in [.6,.1,.1,.1,.8]]
        result = d.score(rows,pred,.5)
        self.assertEqual((result['chips_found'],result['missed'],result['false_alarms']),(1,0,1))
        self.assertEqual(result['clean_group_false_alarm_rate'], 1.)
        self.assertEqual(d.baseline_score(rows)['precision'],1.)
        self.assertIsNone(d.baseline_score([dict(group_id='c', boxes=[])]))

    def test_refuses_shared_identity_across_splits(self):
        rows = [dict(group_id=s, split=s, aliases=['listing:999'], image_sha256=s, file=s) for s in ['train','test']]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'manifest.json'
            path.write_text(json.dumps(dict(schema='whitening-detector-v1',images=rows)))
            with self.assertRaisesRegex(ValueError, 'different splits'):
                d.read_manifest(path)

    @unittest.skipUnless(find_spec('torch') and find_spec('torchvision'), 'Requires the detector environment')
    def test_resume_and_paired_frozen_evaluation(self):
        import torch
        from PIL import Image

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, images, targets=None):
                if targets is not None:
                    return {'loss': self.value.square()}
                return [dict(boxes=torch.tensor([[3.,3.,8.,8.]]), scores=torch.tensor([.014]))]

        with tempfile.TemporaryDirectory() as tmp, patch.object(d, 'make_model', side_effect=lambda *args: Toy()):
            root = Path(tmp)
            rows = []
            for i, split in enumerate(['train','validation','test']):
                path = root/f'{split}.png'
                Image.new('RGB',(16,16),(i,0,0)).save(path)
                rows.append(dict(group_id=split, split=split, file=path.name, boxes=[[3,3,8,8]],
                                 image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(schema='whitening-detector-v1',images=rows)))
            base = ['--dataset', str(manifest), '--device', 'cpu', '--threads','1']
            train = d.main(['train', *base, '--init','random','--out',str(root/'first')])
            self.assertEqual(train['threshold'], .0125)
            first = str(root/'first'/'checkpoint.pt')
            resumed = d.main(['train',*base,'--checkpoint', first,'--out',str(root/'second')])
            self.assertEqual(resumed['completed_epochs'],2)
            second = str(root/'second'/'checkpoint.pt')
            tested = d.main(['evaluate',*base,'--checkpoint',second,'--compare-checkpoint',first,'--out',str(root/'test')])
            self.assertEqual(tested['previous_epoch'],1)
            self.assertEqual(tested['previous_detector']['chips_found'],1)
            self.assertEqual(tested['detector']['chips_found'],1)
            self.assertIn('group:test',tested['evaluated_aliases'])
            with self.assertRaisesRegex(ValueError, 'exact parent'):
                d.main(['evaluate',*base,'--checkpoint',second,'--compare-checkpoint',second,'--out',str(root/'bad')])
