import base64
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pixel_workflow as w
from pixel_whitening import triangle_blue


class PixelWorkflowTests(unittest.TestCase):
    def test_settings_and_no_paid_detection_path(self):
        c=w.settings({'geometry':'triangle','size_pixels':'20'})
        self.assertAlmostEqual(c['measurement_fraction'],20/500*math.sqrt(3)/2)
        inputs=w.detection_inputs(c,'test-output')
        self.assertEqual(inputs['pipeline'],'detection')
        self.assertTrue(inputs['prepare_api_only'])
        self.assertEqual(inputs['max_whitening_cards'],0)
        self.assertEqual(c['candidate_strategy'],'edge_uncertainty')
        self.assertEqual(w.settings({'candidate_strategy':'legacy'})['candidate_strategy'],'legacy')
        for bad in [{'geometry':'oops'},{'size_pixels':'nan'},{'blue_percentage':'0'},
                    {'replay_run_id':'../x'},{'max_listings':0},{'card_ids':'{}'}, {'candidate_strategy':'wrong'}]:
            with self.assertRaises(ValueError):w.settings(bad)

    def test_square_and_circle_dimensions(self):
        mask=np.zeros((100,100),bool);mask[50,50]=True
        hsv=np.full((100,100,3),(110,200,100),np.uint8)
        square=triangle_blue(mask,hsv,'TR',.2,'square')
        circle=triangle_blue(mask,hsv,'TR',.2,'circle')
        self.assertEqual(square['sampled_pixels'],441)
        self.assertLess(circle['sampled_pixels'],square['sampled_pixels'])
        self.assertEqual(circle['blue_fraction'],1)
        self.assertEqual(np.ptp(np.array(square['vertices'])[:,0]),20)

    def test_replay_all_shapes_generates_portable_review_without_fetch(self):
        real_run=subprocess.run
        real_root=w.ROOT
        image=np.full((1050,1010,3),(140,45,20),np.uint8)
        cv2.rectangle(image,(200,220),(203,223),(240,240,240),-1)
        src='data:image/png;base64,'+base64.b64encode(cv2.imencode('.png',image)[1]).decode()
        batch={'cards':[{'id':'test','number':1,'pipeline_decision':{'whitening':'not_run'},
                        'images':[{'label':'Corner close-ups','file':'tips.png','src':src}]},
                       {'id':'missing','number':2,'pipeline_decision':{'whitening':'not_run'},'images':[]}]}
        for shape in ['triangle','square','circle','ring']:
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);audit=root/'replay_batch'/'run'/'_audit';audit.mkdir(parents=True)
                (audit/'run.json').write_text('{}')
                def run(command,**kwargs):
                    self.assertTrue(command[1].endswith('pixel_review.py'),'No listing fetch or API runner allowed')
                    command[1]=str(real_root/'pixel_review.py')
                    return real_run(command,**kwargs,capture_output=True,text=True)
                with patch.object(w,'ROOT',root),patch.object(w,'build_batch',return_value=batch), \
                     patch.object(w.subprocess,'run',side_effect=run), \
                     patch.dict(os.environ,{'PIXEL_TEST_INPUTS':json.dumps({'geometry':shape,'replay_run_id':'123'})}):
                    w.main()
                reviews=list(root.glob('outputs/pixel_tests/*/candidate_review/review.html'))
                self.assertEqual(len(reviews),1)
                self.assertIn('data:image/png;base64,',reviews[0].read_text(encoding='utf-8'))
                data=json.loads(reviews[0].with_name('results.json').read_text())
                self.assertEqual(data[1]['status'],'no_saved_corner_sheet')
                self.assertEqual(data[0]['surrounding_blue_settings']['geometry'],shape)
                self.assertTrue(data[0]['edge_filter_enabled'])

if __name__=='__main__':unittest.main()
