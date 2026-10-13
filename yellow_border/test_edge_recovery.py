import unittest
import copy
from unittest.mock import patch
import numpy as np
from yellow_batch_engine.cardcv.yellow_curves import recover_outer_track,attach_boundary_tracks

class OuterRecoveryTests(unittest.TestCase):
    def test_no_exterior_transition_stays_unresolved(self):
        image=np.full((440,315,3),(35,210,240),np.uint8)
        outer=np.array([[10.,10.],[305.,10.],[305.,430.],[10.,430.]])
        inner=np.array([[22.,22.],[293.,22.],[293.,418.],[22.,418.]])
        self.assertIsNone(recover_outer_track(image,outer,inner,1,295/63,model_mode='forced'))

    def test_known_exterior_is_located_on_the_pixel_transition(self):
        image=np.full((440,315,3),20,np.uint8)
        image[10:430,10:305]=(35,210,240)
        image[22:418,22:293]=(160,100,70)
        outer=np.array([[9.5,9.5],[304.5,9.5],[304.5,429.5],[9.5,429.5]])
        inner=np.array([[21.5,21.5],[292.5,21.5],[292.5,417.5],[21.5,417.5]])
        result=recover_outer_track(image,outer,inner,1,295/63,model_mode='forced')
        self.assertIsNotNone(result)
        self.assertLess(np.max(np.abs(np.array(result['points'])[:,0]-304.5)),.6)
        self.assertGreaterEqual(result['recovery_agreement_count'],1)

    def fixture(self):
        outer=np.array([[10.,10.],[305.,10.],[305.,430.],[10.,430.]])
        inner=np.array([[22.,22.],[293.,22.],[293.,418.],[22.,418.]])
        frame=dict(diag=dict(capture='photo'),result=dict(outer=outer,inner=inner,px_per_mm=295/63))
        def track(img,outer,inner,index,kind,ppm,**kwargs):
            q=outer if kind=='outer' else inner
            a,b=q[index],q[(index+1)%4];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
            points=a+np.linspace(.035,.965,160)[:,None]*v
            return dict(points=points.tolist(),a=a.tolist(),b=b.tolist(),inward=n.tolist(),coefficients=[0.,0.,0.],
                degree=2,support=.95,straight_support=.95,covered_segments=12,counts=[30]*12,
                resampling_stable=True,resampling_max_shift_px=0.,t_range=[.035,.965],
                profile_color_space='bgr',profile_smoothing_mm=0.,recovery_agreement_count=1 if kind=='inner' else 0)
        return np.zeros((440,315,3),np.uint8),frame,track

    def test_accepted_edges_never_enter_recovery(self):
        image,frame,track=self.fixture()
        with patch('yellow_batch_engine.cardcv.yellow_curves.checked_track',side_effect=track), \
             patch('yellow_batch_engine.cardcv.yellow_curves.recover_outer_track') as recovery:
            attach_boundary_tracks(image,frame)
        recovery.assert_not_called()
        self.assertTrue(all(frame['diag']['track_coverage']['outer']))

    def test_new_exterior_cannot_remove_an_accepted_recovered_interior(self):
        image,frame,track=self.fixture();outer,inner=frame['result']['outer'],frame['result']['inner']
        proposed=track(image,outer,inner,1,'outer',295/63)
        proposed['coefficients']=[0.,0.,14.];proposed['recovery_agreement_count']=2
        proposed['points']=(np.array(proposed['points'])+14*np.array(proposed['inward'])).tolist()
        expected=track(image,outer,inner,1,'inner',295/63)['points']
        def checked(img,o,i,index,kind,ppm,**kwargs):
            return None if kind=='outer' and index==1 else track(img,o,i,index,kind,ppm,**kwargs)
        with patch('yellow_batch_engine.cardcv.yellow_curves.checked_track',side_effect=checked), \
             patch('yellow_batch_engine.cardcv.yellow_curves.recover_outer_track',return_value=copy.deepcopy(proposed)):
            attach_boundary_tracks(image,frame)
        self.assertIsNone(frame['diag']['outer_curve_tracks'][1])
        self.assertEqual(frame['diag']['inner_curve_tracks'][1]['points'],expected)
        self.assertTrue(frame['diag']['inner_curve_tracks'][1]['retained'])

if __name__=='__main__':unittest.main()
