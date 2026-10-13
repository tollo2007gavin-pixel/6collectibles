import copy
import unittest
import numpy as np
import cv2
from yellow_batch_engine.frame_diagnostics import evaluate_frames,refit_outer_frame,combine_frame_diagnostics


def fixture(top=0):
    physical=np.float32([[0,0],[63,0],[63,88],[0,88]])
    inner=np.array([[2.7,3.2],[60.7,3.2],[60.7,85.2],[2.7,85.2]])
    actual=np.float32([[200,80],[700,160],[800,1000],[30,980]])
    matrix=cv2.getPerspectiveTransform(physical,actual)
    def project(p):
        h=np.column_stack([p,np.ones(len(p))])@matrix.T
        return h[:,:2]/h[:,2:]
    perturbed=actual+np.array([[3,-2],[-2,3],[4,-3],[-3,2]])
    seed=dict(outer=np.roll(perturbed,top,axis=0).tolist(),inner=np.roll(project(inner),top,axis=0).tolist(),ppm=10.,top_side=top,geometry_reasons=[])
    tracks={}
    for kind,q in [('outer',physical),('inner',inner)]:
        ts=[]
        for i in range(4):
            p=project(q[i]+np.linspace(.04,.96,160)[:,None]*(q[(i+1)%4]-q[i])).tolist()
            ts.append(dict(retained=True,support=1.,points=p,half_resolution_points=copy.deepcopy(p)))
        tracks[kind]=[ts[(i-top)%4] for i in range(4)]
    return seed,tracks,np.roll(actual,top,axis=0)


class FrameDiagnosticTests(unittest.TestCase):
    def test_known_projective_truth_all_card_orientations_and_no_mutation(self):
        for top in range(4):
            seed,tracks,actual=fixture(top);before=copy.deepcopy((seed,tracks))
            r=evaluate_frames(seed,tracks)
            self.assertTrue(np.allclose(r['candidate_frame']['quad'],actual,atol=1e-6))
            for axis,truth in [('lr',54.),('tb',100*3.2/6.)]:
                self.assertAlmostEqual(r['axes'][axis]['candidate_percentage'],truth,places=6)
                self.assertAlmostEqual(r['axes'][axis]['candidate_resampling_shift_pp'],0.,places=8)
                self.assertTrue(all(x>0 for x in r['axes'][axis]['source_border_width_px_p10']))
            self.assertEqual((seed,tracks),before)

    def test_missing_interior_withholds_only_affected_candidate_axis(self):
        seed,tracks,_=fixture();tracks['inner'][2]=None
        r=evaluate_frames(seed,tracks)
        self.assertTrue(r['axes']['lr']['candidate_available'])
        self.assertFalse(r['axes']['tb']['candidate_available'])
        self.assertNotIn('initial_percentage',r['axes']['tb'])

    def test_crossed_bottom_does_not_block_independent_lr_diagnostic(self):
        seed,tracks,_=fixture()
        tracks['inner'][2]['points']=(np.asarray(tracks['outer'][2]['points'])+[0,100]).tolist()
        r=evaluate_frames(seed,tracks)
        self.assertTrue(r['axes']['lr']['candidate_available'])
        self.assertFalse(r['axes']['tb']['candidate_available'])

    def test_missing_outer_prevents_candidate_but_not_supported_original_axis(self):
        seed,tracks,_=fixture();tracks['outer'][0]=None
        r=evaluate_frames(seed,tracks)
        self.assertFalse(r['candidate_frame']['available'])
        self.assertIn('initial_percentage',r['axes']['lr'])
        self.assertFalse(r['axes']['lr']['candidate_available'])

    def test_missing_half_path_is_not_zero_uncertainty(self):
        seed,tracks,_=fixture();del tracks['inner'][3]['half_resolution_points']
        r=evaluate_frames(seed,tracks)['axes']
        self.assertTrue(r['lr']['candidate_available'])
        self.assertNotIn('candidate_resampling_shift_pp',r['lr'])
        self.assertIn('candidate_resampling_shift_pp',r['tb'])

    def test_resampling_decomposition_distinguishes_path_change_from_frame(self):
        seed,tracks,_=fixture()
        tracks['inner'][3]['half_resolution_points']=(np.asarray(tracks['inner'][3]['points'])+[.7,0]).tolist()
        r=evaluate_frames(seed,tracks)['axes']['lr']
        self.assertGreater(abs(r['path_resampling_shift_pp']),.1)
        self.assertAlmostEqual(r['frame_resampling_shift_pp'],0.,places=9)
        self.assertAlmostEqual(r['path_resampling_shift_pp'],r['candidate_resampling_shift_pp'],places=9)

    def test_candidate_rejects_far_corners_and_degenerate_tracks(self):
        seed,tracks,_=fixture();seed['ppm']=.01
        with self.assertRaises(ValueError):refit_outer_frame(seed,tracks)
        seed,tracks,_=fixture();tracks['outer'][1]['points']=[[1,2],[1,2]]
        self.assertFalse(evaluate_frames(seed,tracks)['candidate_frame']['available'])

    def test_aggregation_uses_qualified_provenance_and_never_cancels_shifts(self):
        def axis(shift,resolved=True):return dict(resolved=resolved,frame_diagnostics=dict(candidate_available=True,candidate_percentage=51.,frame_difference_pp=1.,candidate_resampling_shift_pp=shift))
        r=combine_frame_diagnostics(dict(adaptive=axis(.5),forced=axis(-.5)))
        self.assertEqual(r['max_abs_candidate_resampling_shift_pp'],.5)
        r=combine_frame_diagnostics(dict(adaptive=axis(.2),forced=axis(4.,False)))
        self.assertEqual(r['max_abs_candidate_resampling_shift_pp'],.2)
        self.assertEqual(r['contributing_methods'],['adaptive'])
        r=combine_frame_diagnostics(dict(adaptive=axis(.2,False),forced=axis(.3,False)))
        self.assertFalse(r['baseline_qualified']);self.assertTrue(r['advisory_only'])

    def test_opt_in_selects_candidate_without_bypassing_physical_gates(self):
        from yellow_batch_engine.compare import measure_seed_tracks
        seed,tracks,_=fixture();image=np.zeros((1100,900,3),np.uint8)
        initial=measure_seed_tracks(image,seed,tracks)
        candidate=measure_seed_tracks(image,seed,tracks,measurement_frame='refit_when_supported')
        self.assertNotAlmostEqual(initial['axes']['lr']['percentage'],54.,places=3)
        self.assertAlmostEqual(candidate['axes']['lr']['percentage'],54.,places=6)
        self.assertEqual(candidate['frame_diagnostics']['active_frame'],'refit')
        self.assertEqual(candidate['paths'],initial['paths'])
        seed['geometry_reasons']=['fitted perspective 0.193 > 0.100']
        rejected=measure_seed_tracks(image,seed,tracks,measurement_frame='refit_when_supported')
        self.assertFalse(rejected['axes']['lr']['resolved'])
        self.assertIsNone(rejected['axes']['lr']['percentage'])
        self.assertTrue(rejected['axes']['lr']['frame_diagnostics']['candidate_available'])

    def test_opt_in_falls_back_when_other_outer_side_missing(self):
        from yellow_batch_engine.compare import measure_seed_tracks
        seed,tracks,_=fixture();tracks['outer'][0]=None
        r=measure_seed_tracks(np.zeros((1100,900,3),np.uint8),seed,tracks,measurement_frame='refit_when_supported')
        self.assertTrue(r['axes']['lr']['resolved'])
        self.assertEqual(r['axes']['lr']['frame_diagnostics']['active_frame'],'initial')
        self.assertFalse(r['axes']['tb']['resolved'])

    def test_csv_does_not_publish_diagnostic_percentages_for_rejected_axes(self):
        import tempfile,csv
        from pathlib import Path
        from ebay_line_fit_test import checkpoint
        diag=dict(candidate_available=True,candidate_percentage=52.,frame_difference_pp=1.,active_frame='initial',max_abs_candidate_resampling_shift_pp=.4)
        result=dict(status='Review / unresolved',axes={
            'lr':dict(resolved=False,reasons=['perspective'],frame_diagnostics=diag),
            'tb':dict(resolved=True,percentage=51.,reasons=[],frame_diagnostics=diag)})
        batch=dict(records=[dict(title='fixture',query='fixture',item_id='1',overlays={'combined':'fixture.jpg'},combined=result)])
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint(Path(tmp),batch)
            with (Path(tmp)/'measurements.csv').open(encoding='utf-8-sig') as f:r=next(csv.DictReader(f))
        self.assertEqual(r['left_pct'],'');self.assertEqual(r['candidate_frame_left_pct'],'')
        self.assertEqual(float(r['candidate_frame_top_pct']),52.)


if __name__=='__main__':unittest.main()
