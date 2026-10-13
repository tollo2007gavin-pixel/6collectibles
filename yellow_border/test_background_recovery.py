"""Background cleanup must not change usable results or bypass quality gates."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from yellow_batch_engine.background_recovery import should_attempt,recover_background,_opened
from yellow_batch_engine.cardcv.pipeline import DEFAULT_GATES


def rejected(reasons=('inner frame is outside the physical card',)):
    return dict(seeds=[dict(geometry_reasons=list(reasons))],image_gate=dict(fail=False),
        methods={m:dict(axes={a:dict(resolved=False,reasons=list(reasons)) for a in ('lr','tb')}) for m in ('adaptive','forced')})


class BackgroundRecoveryTests(unittest.TestCase):
    def test_existing_axis_is_never_relocalized(self):
        analysis=rejected();analysis['methods']['forced']['axes']['tb']['resolved']=True
        with patch('yellow_batch_engine.background_recovery.analyze') as detector:
            self.assertFalse(should_attempt(analysis))
            self.assertIs(recover_background(None,analysis),analysis)
            detector.assert_not_called()

    def test_hard_crop_failure_is_never_relocalized(self):
        analysis=rejected();analysis['image_gate']['fail']=True
        with patch('yellow_batch_engine.background_recovery.analyze') as detector:
            self.assertIs(recover_background(None,analysis),analysis)
            detector.assert_not_called()

    def test_perspective_only_or_edge_support_only_is_not_background(self):
        for reasons in [[],['fitted perspective 0.25 > 0.10']]:
            self.assertFalse(should_attempt(rejected(reasons)))

    def test_opening_detaches_thin_pattern_without_editing_source(self):
        image=np.zeros((400,400,3),np.uint8)
        image[60:340,100:300]=255
        image[190:192,20:100]=255
        image[100:290,20:22]=255
        saved=image.copy()
        mask=_opened(lambda im:im[:,:,0].copy(),.01)(image)
        self.assertTrue(np.array_equal(image,saved))
        self.assertEqual(mask[190,70],0)
        self.assertEqual(mask[200,200],255)

    def test_one_mask_variant_cannot_establish_recovery(self):
        analysis=rejected();candidate=dict(method='one',ok=True,centering=dict(left_pct=50.,top_pct=50.))
        with patch('yellow_batch_engine.background_recovery.analyze',return_value=(candidate,[candidate])),patch('yellow_batch_engine.background_recovery.rejection_reasons',return_value=[]):
            after=recover_background(None,analysis)
        self.assertFalse(after['background_recovery']['applied'])
        self.assertEqual(after['methods'],analysis['methods'])

    def test_recovered_geometry_does_not_override_perspective_policy(self):
        image=np.full((440,315,3),20,np.uint8)
        image[10:430,10:305]=(35,210,240);image[22:418,22:293]=(160,100,70)
        outer=np.array([[9.5,9.5],[304.5,9.5],[304.5,429.5],[9.5,429.5]])
        inner=np.array([[21.5,21.5],[292.5,21.5],[292.5,417.5],[21.5,417.5]])
        candidate=dict(method='first',ok=True,centering=dict(left_pct=50.,top_pct=50.),
            result=dict(outer=outer,inner=inner,px_per_mm=295/63),diag=dict(keystone=.18))
        second=dict(candidate,method='second');analysis=rejected()
        def reasons(candidate,gates,meta=None):
            return ['fitted perspective 0.180 > 0.100'] if gates['keystone']<.18 else []
        with patch.dict(DEFAULT_GATES,keystone=.1),patch('yellow_batch_engine.background_recovery.analyze',return_value=(candidate,[candidate,second])),patch('yellow_batch_engine.background_recovery.rejection_reasons',side_effect=reasons),patch('yellow_batch_engine.background_recovery.card_orientation',return_value=(0,{})):
            after=recover_background(image,analysis)
        self.assertTrue(after['background_recovery']['applied'])
        for method in after['methods'].values():
            for value in method['axes'].values():
                self.assertFalse(value['resolved'])
                self.assertIsNone(value.get('percentage'))
                self.assertEqual(value['reasons'],['fitted perspective 0.180 > 0.100'])
        self.assertNotIn('background_recovery',analysis)

if __name__=='__main__':unittest.main()
