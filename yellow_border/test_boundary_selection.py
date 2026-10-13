"""Known-geometry checks for wider yellow strips competing with printed lines."""
import unittest
from unittest.mock import patch
import numpy as np
import cv2
from yellow_batch_engine.cardcv.enumerate_edges import analyze_side,refine_by_enumeration
from yellow_batch_engine.cardcv.centering import centering


class BoundarySelectionTests(unittest.TestCase):
    def fixture(self,width=38,rotation=0,background=(20,20,20)):
        image=np.full((240,640,3),background,np.uint8)
        image[120:120+width]=(30,190,220)
        image[120+width:]=(170,70,20)
        # Printed transitions give the old nominal-width search nonempty junk
        # pairs even when the physical yellow strip is wider than 3.4 mm.
        image[176:183]=(210,210,210)
        image[195:200]=(70,40,20)
        a=np.array([20.,119.5]);b=np.array([620.,119.5]);n=np.array([0.,-1.])
        for _ in range(rotation):
            w=image.shape[1];image=np.rot90(image).copy()
            a=np.array([a[1],w-1-a[0]]);b=np.array([b[1],w-1-b[0]]);n=np.array([n[1],-n[0]])
        return image,a,b,n

    def analyze(self,*args,**kwargs):
        im,a,b,n=self.fixture(*args,**kwargs)
        return analyze_side(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32),cv2.cvtColor(im,cv2.COLOR_BGR2HSV),
            a,b,n,10.,30.,meta=dict(retry_wide_pairs=True),lab_img=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32))

    def test_true_wide_strip_survives_narrow_printed_alternatives(self):
        side=self.analyze()
        self.assertIsNotNone(side)
        self.assertTrue(any('width_recovery' not in p for p in side['pairs']))
        best=side['pairs'][0]
        self.assertIn('width_recovery',best)
        self.assertAlmostEqual(best['outer']['mean_d'],0.,delta=.06)
        self.assertAlmostEqual(best['inner']['mean_d'],3.8,delta=.06)
        self.assertGreaterEqual(best['yellow_in'],.85)
        self.assertLessEqual(best['yellow_out'],.25)

    def test_same_evidence_works_on_each_side_orientation(self):
        for rotation in (1,2,3):
            side=self.analyze(rotation=rotation)
            self.assertIn('width_recovery',side['pairs'][0])
            self.assertAlmostEqual(side['pairs'][0]['width_mm'],3.8,delta=.08)

    def test_range_of_unequal_widths_is_not_pulled_toward_nominal(self):
        for width in (35,38,41):
            side=self.analyze(width=width)
            self.assertIn('width_recovery',side['pairs'][0])
            self.assertAlmostEqual(side['pairs'][0]['width_mm'],width/10,delta=.08)

    def test_supported_nominal_width_does_not_trigger_wide_retry(self):
        side=self.analyze(width=22)
        self.assertFalse(any('width_recovery' in p for p in side['pairs']))
        self.assertAlmostEqual(side['pairs'][0]['width_mm'],2.2,delta=.08)

    def test_yellow_background_cannot_support_the_added_exterior(self):
        side=self.analyze(background=(30,190,220))
        self.assertTrue(side is None or not any('width_recovery' in p for p in side['pairs']))

    def test_width_retry_is_bounded(self):
        side=self.analyze(width=45)
        self.assertTrue(side is None or not any('width_recovery' in p for p in side['pairs']))

    def test_unequal_borders_keep_their_known_centering(self):
        image=np.full((940,690,3),20,np.uint8)
        image[20:900,20:650]=(30,190,220)
        image[42:862,44:626]=(170,70,20)
        image[832:840,44:626]=(210,210,210)
        image[814:820,44:626]=(70,40,20)
        outer=np.array([[19.5,19.5],[649.5,19.5],[649.5,899.5],[19.5,899.5]])
        result=refine_by_enumeration(image,outer,meta=dict(capture='photo',holder='unknown',retry_wide_pairs=True))
        self.assertIsNotNone(result)
        measurement=centering(result['outer'],result['inner'],portrait=True)
        self.assertAlmostEqual(measurement['top_pct'],100*22/60,delta=.5)
        self.assertAlmostEqual(measurement['left_pct'],50.,delta=.5)


class RecoveryGuardTests(unittest.TestCase):
    def rejected(self,reasons=('inner frame is outside the physical card',)):
        return dict(seeds=[dict(geometry_reasons=list(reasons))],image_gate=dict(fail=False),
            methods={m:dict(axes={a:dict(resolved=False,reasons=list(reasons)) for a in ('lr','tb')}) for m in ('adaptive','forced')})

    def test_existing_axes_hard_failures_and_perspective_are_preserved(self):
        from yellow_batch_engine.boundary_recovery import recover_boundary_selection
        usable=self.rejected();usable['methods']['forced']['axes']['lr']['resolved']=True
        cropped=self.rejected();cropped['image_gate']['fail']=True
        for analysis in [usable,cropped,self.rejected([]),self.rejected(['fitted perspective 0.25 > 0.10'])]:
            with patch('yellow_batch_engine.boundary_recovery.analyze') as detector:
                self.assertIs(recover_boundary_selection(None,analysis),analysis)
                detector.assert_not_called()

    def test_unrelated_frame_and_single_localizer_cannot_establish_recovery(self):
        from yellow_batch_engine.boundary_recovery import recover_boundary_selection
        for recovered in [[],[dict(side=2)]]:
            analysis=self.rejected()
            candidate=dict(method='one',ok=True,centering=dict(left_pct=50.,top_pct=50.),result=dict(pair_width_recovery=recovered))
            with patch('yellow_batch_engine.boundary_recovery.analyze',return_value=(candidate,[candidate])),patch('yellow_batch_engine.boundary_recovery.rejection_reasons',return_value=[]):
                after=recover_boundary_selection(None,analysis)
            self.assertFalse(after['boundary_selection_recovery']['applied'])
            self.assertEqual(after['methods'],analysis['methods'])

    def test_recovered_geometry_respects_configured_perspective_policy(self):
        from yellow_batch_engine.boundary_recovery import recover_boundary_selection,DEFAULT_GATES
        image=np.full((440,315,3),20,np.uint8)
        image[10:430,10:305]=(35,210,240);image[22:418,22:293]=(160,100,70)
        outer=np.array([[9.5,9.5],[304.5,9.5],[304.5,429.5],[9.5,429.5]])
        inner=np.array([[21.5,21.5],[292.5,21.5],[292.5,417.5],[21.5,417.5]])
        candidate=dict(method='first',ok=True,centering=dict(left_pct=50.,top_pct=50.),
            result=dict(outer=outer,inner=inner,px_per_mm=295/63,pair_width_recovery=[dict(side=2)]),diag=dict(keystone=.18))
        second=dict(candidate,method='second');analysis=self.rejected()
        def reasons(candidate,gates,meta=None):
            return ['fitted perspective 0.180 > 0.100'] if gates['keystone']<.18 else []
        with patch.dict(DEFAULT_GATES,keystone=.1),patch('yellow_batch_engine.boundary_recovery.analyze',return_value=(candidate,[candidate,second])),patch('yellow_batch_engine.boundary_recovery.rejection_reasons',side_effect=reasons),patch('yellow_batch_engine.boundary_recovery.card_orientation',return_value=(0,{})):
            after=recover_boundary_selection(image,analysis)
        self.assertTrue(after['boundary_selection_recovery']['applied'])
        for method in after['methods'].values():
            for value in method['axes'].values():
                self.assertFalse(value['resolved'])
                self.assertIsNone(value.get('percentage'))
                self.assertEqual(value['reasons'],['fitted perspective 0.180 > 0.100'])
        self.assertNotIn('boundary_selection_recovery',analysis)


if __name__=='__main__':unittest.main()
