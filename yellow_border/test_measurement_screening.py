import unittest
from yellow_batch_engine.measurement_screening import screen_axis

class MeasurementScreeningTests(unittest.TestCase):
    def test_requested_example_includes_exact_endpoint_and_complement(self):
        r=screen_axis(53.,.5,max_side_pct=52.5)
        self.assertTrue(r['accepted']);self.assertEqual(r['status'],'accepted_due_to_uncertainty')
        self.assertEqual(r['percentage_range'],[52.5,53.5])
        self.assertEqual(r['opposite_percentage_range'],[46.5,47.5])
        self.assertEqual(r['possible_passing_range'],[52.5,52.5])
        self.assertIsNone(r['statistical_confidence_level'])

    def test_gate_is_symmetric_for_opposite_heavy_margin(self):
        self.assertTrue(screen_axis(47.,.5,max_side_pct=52.5)['accepted'])
        self.assertFalse(screen_axis(46.9,.5,max_side_pct=52.5)['accepted'])
        self.assertFalse(screen_axis(53.1,.5,max_side_pct=52.5)['accepted'])

    def test_range_and_point_pass_are_distinguished(self):
        self.assertEqual(screen_axis(51.,.5,max_side_pct=52.5)['status'],'accepted_within_estimated_range')
        self.assertEqual(screen_axis(52.4,.5,max_side_pct=52.5)['status'],'accepted_point_estimate')

    def test_image_quality_gate_takes_precedence_even_with_large_range(self):
        r=screen_axis(60.,20.,max_side_pct=52.5,measurement_qualified=False)
        self.assertFalse(r['accepted']);self.assertEqual(r['status'],'withheld_unqualified_measurement')

    def test_unknown_uncertainty_is_not_assumed_zero(self):
        self.assertEqual(screen_axis(50.,None,max_side_pct=52.5)['status'],'withheld_unavailable_uncertainty')
        self.assertTrue(screen_axis(50.,0.,max_side_pct=52.5)['accepted'])

    def test_invalid_inputs_fail_explicitly(self):
        for percentage,error,gate in [(101.,.5,52.5),(50.,-.5,52.5),(50.,.5,49.),(float('nan'),.5,52.5),(50.,float('inf'),52.5)]:
            with self.assertRaises(ValueError):screen_axis(percentage,error,max_side_pct=gate)

if __name__=='__main__':unittest.main()
