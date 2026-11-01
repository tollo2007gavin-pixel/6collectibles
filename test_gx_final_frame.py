import copy,unittest
from unittest.mock import patch
import cv2,numpy as np
from gx_border.gx_guarded_engine import modern
from gx_border.gx_guarded_engine.final_frame import frame_candidate,repair_certificate,certify_frame,reobserve_curve
from gx_border.gx_guarded_engine.source_completion import extend_outer
from gx_border.gx_guarded_engine.holder_topology import FRAME
from gx_border.gx_guarded_engine.tracks import SIDES

def fixture():
    q=np.float32([[80,80],[1340,80],[1340,1840],[80,1840]])
    g=dict(outer=q.tolist(),inner=(q+np.array([[45,45],[-45,45],[-45,-45],[45,-45]])).tolist())
    tracks={kind:{s:dict(supported=True,coverage=.8,degree=1,coefficients=[9999,9999],
                         segments=[(np.asarray(g[kind])[k]+np.linspace(.1,.9,161)[:,None]*(np.asarray(g[kind])[(k+1)%4]-np.asarray(g[kind])[k])).tolist()])
                  for k,s in enumerate(SIDES)} for kind in ['outer','inner']}
    return np.zeros((1920,1420,3),np.uint8),g,tracks

class FinalFrameTests(unittest.TestCase):
    def test_coefficients_rebuilt_from_source_not_stale_metadata(self):
        im,g,t=fixture();c=frame_candidate(t['outer']['top'],g,0)
        self.assertLess(np.max(np.abs(c['track']['coefficients'])),.001)
        self.assertEqual(t['outer']['top']['coefficients'],[9999,9999])

    def test_repair_sequence_cannot_skip_stability(self):
        im,g,t=fixture();r=dict(geometry=g,tracks=t,repairs=[dict(side='top')])
        self.assertFalse(repair_certificate(im,r)['verified'])

    def test_blank_photo_cannot_certify_a_drawn_frame(self):
        im,g,t=fixture();self.assertFalse(certify_frame(im,g,t,dict(issues=[]))['verified'])
        self.assertFalse(reobserve_curve(im,g,t['outer']['top'],0)['verified'])

    def test_unavailable_outer_does_not_release_other_axis(self):
        im,g,t=fixture();t['outer']['right']['supported']=False
        self.assertFalse(certify_frame(im,g,t,dict(issues=[]))['verified'])

    def test_quality_issue_not_cleared_by_final_frame(self):
        im,g,t=fixture();self.assertFalse(certify_frame(im,g,t,dict(issues=['unreliable_perspective']))['verified'])

    def test_earlier_rounded_edge_blocks_new_frame_certification(self):
        im,g,t=fixture();e=dict(hypotheses={s:[] for s in SIDES})
        e['hypotheses']['bottom']=[dict(offset=-15,corner_score=2,topology_score=1,track=dict(coverage=.8))]
        proof=certify_frame(im,g,t,dict(issues=[]),e)
        self.assertFalse(proof['verified']);self.assertEqual(proof['outward_candidates'][0]['side'],'bottom')

    def test_weak_outward_seam_is_not_counted_as_rounded_competitor(self):
        im,g,t=fixture();e=dict(hypotheses={s:[] for s in SIDES})
        e['hypotheses']['bottom']=[dict(offset=-15,corner_score=2,topology_score=.1,track=dict(coverage=.8))]
        proof=certify_frame(im,g,t,dict(issues=[]),e)
        self.assertFalse(proof['verified']);self.assertNotIn('outward_candidates',proof)

    def test_distant_background_track_is_not_a_gx_shoulder_competitor(self):
        im,g,t=fixture();e=dict(hypotheses={s:[] for s in SIDES})
        e['hypotheses']['right']=[dict(offset=-70,corner_score=3,topology_score=1,track=dict(coverage=.8))]
        proof=certify_frame(im,g,t,dict(issues=[]),e)
        self.assertFalse(proof['verified']);self.assertNotIn('outward_candidates',proof)

    def test_outer_extension_preserves_supported_full_tracks(self):
        im,g,t=fixture();new,changes=extend_outer(im,g,t)
        self.assertEqual(new,t);self.assertEqual(changes,[])

    def test_blank_pixels_do_not_extend_short_observation(self):
        im,g,t=fixture();t['outer']['right']['coverage']=.22
        new,changes=extend_outer(im,g,t);self.assertEqual(new,t);self.assertEqual(changes,[])

    def test_hard_failure_never_enters_recovery(self):
        im,g,t=fixture();base=dict(status='hard_failure',axes={},reasons=['missing exterior'])
        with patch.object(modern,'seed_analyze',return_value=base),patch.object(modern,'reviewed_evidence',return_value=None),patch('gx_border.gx_guarded_engine.source_completion.recover_frame') as recover:
            result=modern.analyze(im);self.assertEqual(result['status'],'hard_failure');recover.assert_not_called()

    def test_ambiguous_frame_must_reconcile_candidates_before_refine(self):
        im,g,t=fixture();base=dict(status='unresolved',axes={},reasons=[],diagnostic_geometry=g,localization=dict(ambiguous=True),localization_candidates=[g])
        with patch.object(modern,'seed_analyze',return_value=base),patch.object(modern,'reviewed_evidence',return_value=None),patch('gx_border.gx_guarded_engine.source_completion.recover_frame',return_value=(None,dict(reason='conflicting source tracks'))) as recover,patch.object(modern,'refine') as refine:
            result=modern.analyze(im);self.assertEqual(result['status'],'unresolved');recover.assert_called_once();refine.assert_not_called()

if __name__=='__main__':unittest.main()
