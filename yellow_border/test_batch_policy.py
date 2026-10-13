import copy,unittest
from unittest.mock import patch
from yellow_batch_engine.batch_policy import analyze_front,describe_measurement,DEFAULT_GATES


class BatchPolicyTests(unittest.TestCase):
    def fixture(self,score=.18):
        analysis=dict(methods={m:dict(axes={a:dict(resolved=True,selected_trial=0) for a in ('lr','tb')},trials=[dict(seed=dict(keystone=score))]) for m in ('adaptive','forced')})
        combined=dict(status='Measured',axes={a:dict(resolved=True,percentage=53.) for a in ('lr','tb')})
        return analysis,combined

    def test_gate_is_scoped_and_rebuilt_frame_requested(self):
        old=DEFAULT_GATES['keystone']
        def check(image,**kwargs):
            self.assertEqual(DEFAULT_GATES['keystone'],.2)
            self.assertEqual(kwargs['measurement_frame'],'refit_when_supported')
            return {'ok':True}
        with patch('yellow_batch_engine.batch_policy.compare',side_effect=check):
            self.assertEqual(analyze_front(None,measurement_frame='refit_when_supported',perspective_limit=.2),{'ok':True})
        self.assertEqual(DEFAULT_GATES['keystone'],old)
        with patch('yellow_batch_engine.batch_policy.compare',side_effect=ValueError('failure')):
            with self.assertRaises(ValueError):analyze_front(None,measurement_frame='initial',perspective_limit=.2)
        self.assertEqual(DEFAULT_GATES['keystone'],old)

    def test_review_band_inclusive_limits_and_no_selection_gate(self):
        for score,category,allowance in [(.15,'pass',0.),(.150001,'pass/review',1.),(.2,'pass/review',1.)]:
            a,c=self.fixture(score);saved=copy.deepcopy(c)
            p=describe_measurement(a,c)
            self.assertEqual(p['category'],category)
            self.assertEqual(p['axis_allowance_pp']['lr'],allowance)
            self.assertEqual(p['axis_percentage_range']['lr'],[53-allowance,53+allowance])
            self.assertFalse(p['centering_selection_gate_applied'])
            self.assertEqual(c,saved)

    def test_only_qualified_axis_methods_contribute(self):
        a,c=self.fixture(.1)
        a['methods']['forced']['trials'][0]['seed']['keystone']=.19
        a['methods']['forced']['axes']['tb']['resolved']=False
        p=describe_measurement(a,c)
        self.assertEqual(p['axis_allowance_pp'],dict(lr=1.,tb=0.))

    def test_missing_axis_and_hard_fail_never_become_pass(self):
        a,c=self.fixture()
        c['axes']['tb']['resolved']=False
        p=describe_measurement(a,c)
        self.assertEqual(p['category'],'undetermined')
        self.assertIsNone(p['axis_allowance_pp']['tb'])
        c['status']='Fail'
        self.assertEqual(describe_measurement(a,c)['category'],'fail')

if __name__=='__main__':unittest.main()
