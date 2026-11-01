import unittest,copy,numpy as np
from unittest.mock import patch
from gx_border.gx_guarded_engine import final_frame,sparse_keyline
from test_gx_final_frame import fixture


class CandidateIdentityTests(unittest.TestCase):
    def test_resize_audit_maps_pixel_centres_and_original_stays_strict(self):
        im,g,t=fixture();calls=[]
        def observed(image,geometry,track,k,**kw):calls.append((image.shape,geometry,kw));return dict(verified=True)
        with patch.object(final_frame,'reobserve_curve',side_effect=observed):final_frame.audit_curve(im,g,t['outer']['top'],0)
        np.testing.assert_allclose(calls[1][1]['outer'],np.asarray(g['outer'])*.75-.125)
        self.assertEqual([c[2]['resampling_uncertainty_px'] for c in calls],[0.,.5,0.])

    def test_supported_inner_tracks_are_never_replaced(self):
        im,g,t=fixture()
        with patch.object(sparse_keyline,'select') as select:
            updated,changes=sparse_keyline.recover(im,g,t)
        select.assert_not_called();self.assertEqual(updated,t);self.assertFalse(changes)

    def test_more_coverage_without_physical_sequence_cannot_replace_inner(self):
        im,g,t=fixture();t['inner']['bottom']['coverage']=.3;c=copy.deepcopy(t['inner']['bottom']);c['coverage']=.8
        with patch.object(sparse_keyline,'select',return_value=(c,{})),patch.object(sparse_keyline,'audit_sequence',return_value=dict(verified=False)):
            updated,changes=sparse_keyline.recover(im,g,t)
        self.assertEqual(updated,t);self.assertFalse(changes)

    def test_unstable_recovered_keyline_retains_previous_track(self):
        im,g,t=fixture();t['inner']['bottom']['coverage']=.3;c=copy.deepcopy(t['inner']['bottom']);c['coverage']=.8
        with patch.object(sparse_keyline,'select',side_effect=[(c,{}),(None,{}),(None,{})]),patch.object(sparse_keyline,'audit_sequence',return_value=dict(verified=True)):
            updated,changes=sparse_keyline.recover(im,g,t)
        self.assertEqual(updated,t);self.assertFalse(changes)

    def test_competing_inner_modes_do_not_choose_by_coverage_alone(self):
        im,g,t=fixture();a=t['inner']['bottom'];b=copy.deepcopy(a);b['segments']=[(np.asarray(s)+[0,15]).tolist() for s in b['segments']]
        with patch.object(sparse_keyline,'inner_from_physical_edge',return_value=[a,b]):candidate,proof=sparse_keyline.select(im,g['outer'],2)
        self.assertIsNone(candidate);self.assertEqual(proof['reason'],'competing inward GX signatures')

if __name__=='__main__':unittest.main()
