"""Known-geometry checks for interior recovery and observed-span handling."""
import unittest
import numpy as np
from yellow_batch_engine.cardcv.yellow_curves import checked_track,retention_reasons,recover_inner_track
from yellow_batch_engine.cardcv.photo_margins import boundary_margins


class InteriorRecoveryTests(unittest.TestCase):
    def test_corner_print_interruptions_recover_without_inventing_endpoints(self):
        image=np.full((880,630,3),20,np.uint8)
        image[20:860,20:610]=(35,210,240)
        image[44:836,44:586]=(160,100,70)
        # A printed label obscures the left transition near both ends, while
        # the middle still has a precisely known physical position at x=43.5.
        image[44:124,33:100]=(220,220,220)
        image[756:836,33:100]=(220,220,220)
        outer=np.array([[19.5,19.5],[609.5,19.5],[609.5,859.5],[19.5,859.5]])
        inner=np.array([[43.5,43.5],[585.5,43.5],[585.5,835.5],[43.5,835.5]])
        ppm=590/63
        primary=checked_track(image,outer,inner,3,'inner',ppm,model_mode='forced')
        self.assertTrue(retention_reasons(primary,ppm,allow_straight=True))
        recovered=recover_inner_track(image,outer,inner,3,ppm,model_mode='forced')
        self.assertIsNotNone(recovered)
        self.assertLess(np.max(np.abs(np.array(recovered['points'])[:,0]-43.5)),.6)
        self.assertEqual(recovered['t_range'],[.09,.91])
        self.assertGreaterEqual(recovered['recovery_agreement_count'],1)

    def test_absent_transition_stays_unresolved(self):
        image=np.full((440,315,3),(35,210,240),np.uint8)
        outer=np.array([[10.,10.],[305.,10.],[305.,430.],[10.,430.]])
        inner=np.array([[22.,22.],[293.,22.],[293.,418.],[22.,418.]])
        self.assertIsNone(recover_inner_track(image,outer,inner,0,295/63,model_mode='forced'))

    def test_measurement_uses_common_span_without_extrapolation(self):
        outer=np.array([[0.,0.],[63.,0.],[63.,88.],[0.,88.]])
        inner=np.array([[3.,4.],[60.,4.],[60.,84.],[3.,84.]])
        paths={}
        for kind,q in [('outer',outer),('inner',inner)]:
            paths[kind]=[]
            for i in range(4):
                trim=.09 if kind=='inner' and i==3 else .035
                t=np.linspace(trim,1-trim,160)[:,None]
                paths[kind].append((q[i]+t*(q[(i+1)%4]-q[i])).tolist())
        result=boundary_margins(outer,inner,paths['outer'],paths['inner'],0)
        self.assertAlmostEqual(result['ratios']['left_pct'],50.)
        self.assertAlmostEqual(result['ratios']['top_pct'],50.)
        low,high=result['axes']['lr']['sampled_fraction']
        self.assertGreaterEqual(low,(4+.09*80)/88)
        self.assertLessEqual(high,(4+.91*80)/88)


if __name__=='__main__':unittest.main()
