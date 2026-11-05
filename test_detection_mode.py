import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
import card_runner as r
from detection_replay import candidates, workflow_inputs


class DetectionTests(unittest.TestCase):
    def test_detection_bypasses_centering_and_paid_calls_even_with_zero_cap(self):
        for ready in (True, False):
            s=r.Settings(pipeline='detection', max_whitening_cards=0)
            row=dict(item_id='v1|123|0',card_id='22',border_type='yellow',title='Test')
            packet=dict(ready=ready,reason='missing',context='original')
            paid=Mock(side_effect=AssertionError('Paid call forbidden'))
            with patch.object(r,'centering_gate',side_effect=AssertionError('Centering forbidden')):
                r.finish_listing(row,{},None,None,s,None,r.new_state(s),lambda:None,preparer=Mock(return_value=packet),classifier=paid)
            paid.assert_not_called()
            self.assertEqual(row['whitening']['decision'],'not_run')
            self.assertEqual(row['detection']['sheet_ready'],ready)
            self.assertFalse(row['accepted'])

    def test_baseline_makes_sheets_without_stricter_detector_and_replays_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);audit=out/'_audit';audit.mkdir()
            pixels=np.full((900,900,3),205,np.uint8)
            cv2.rectangle(pixels,(220,160),(620,720),(150,30,15),-1)
            raw=cv2.imencode('.png',pixels)[1].tobytes()
            client=Mock();client.download.return_value=raw
            row=dict(item_id='v1|123|0',card_id='22',border_type='yellow',title='Test',image_url='test',url='')
            with patch('whitening_geometry.detect',side_effect=AssertionError('Wrong detector')):
                packet=r.prepare_back(client,{},row,audit,r.Settings(pipeline='detection'))
            self.assertTrue(packet['ready'])
            self.assertTrue(Path(packet['sheet']).is_file())
            manifest=json.loads(Path(packet['manifest']).read_text())
            self.assertEqual(len(manifest['photos'][0]['detected_quad']),4)
            self.assertTrue((Path(packet['manifest']).parent/manifest['photos'][0]['boundary_overlay']).is_file())
            row['back_packet']=packet
            (audit/'run.json').write_text(json.dumps({'records':[row]}))
            replay=list(candidates(out))
            self.assertEqual(replay[0][0]['item_id'],row['item_id'])
            self.assertEqual(Path(replay[0][1]['image']['imageUrl']).read_bytes(),raw)

    def test_workflow_mapping(self):
        self.assertEqual(workflow_inputs({'pipeline':'detection','replay_run_id':'123'})['replay_folder'],'replay_batch')
        self.assertNotIn('replay_run_id',workflow_inputs({'pipeline':'api','replay_run_id':''}))
        for p in ({'pipeline':'api','replay_run_id':'123'},{'pipeline':'detection','replay_run_id':'../x'}):
            with self.assertRaises(ValueError):workflow_inputs(p)

    def test_detection_run_needs_no_reference_bank_and_keeps_all_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=r.Settings(pipeline='detection',output_mode='production',output_root=tmp,max_listings=1)
            meta=dict(item_id='v1|123|0',card_id='22',border_type='yellow',title='Test',url='')
            with patch.object(r,'load_bank',side_effect=AssertionError('No references')),patch.object(r,'ebay_candidates',return_value=iter([(meta,{})])):
                r.run(s,client_factory=Mock,processor=Mock(side_effect=AssertionError('No centering')),preparer=Mock(return_value={'ready':False,'reason':'no_photo'}),classifier=Mock(side_effect=AssertionError('No API')))
            state=json.loads(next(Path(tmp).glob('*/_audit/run.json')).read_text())
            self.assertEqual(len(state['records']),1)
            self.assertEqual(state['settings']['output_mode'],'testing')
            self.assertEqual(state['whitening_attempts'],0)

if __name__=='__main__':unittest.main()
