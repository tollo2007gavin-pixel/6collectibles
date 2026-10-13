import copy
import unittest
import numpy as np
import cv2
from test_pale_border import fixture
from yellow_batch_engine.yellow_validation import validate_yellow_illumination
from yellow_batch_engine.cardcv.yellow_precision import frame_evidence


class IlluminationTests(unittest.TestCase):
    def illuminated_fixture(self):
        image,seed,tracks=fixture();image[20:30,20:180]=(160,211,220)
        seed['geometry_reasons']=['chroma_dev 0.70 > 0.50']
        return image,seed,tracks

    def test_yellow_hue_survives_large_saturation_change(self):
        image,seed,tracks=fixture()
        image[20:30,20:180]=(160,211,220)
        seed['geometry_reasons']=['chroma_dev 0.70 > 0.50']
        original=copy.deepcopy((seed,tracks))
        result=validate_yellow_illumination(image,seed,tracks)
        self.assertTrue(result['applied'],result)
        self.assertEqual((seed,tracks),original)

    def test_neutral_holder_and_different_hue_do_not_qualify(self):
        for colour in [(200,200,200),(210,90,210)]:
            image,seed,tracks=fixture();image[20:30,20:180]=colour
            seed['geometry_reasons']=['chroma_dev 0.70 > 0.50']
            self.assertFalse(validate_yellow_illumination(image,seed,tracks)['applied'])

    def test_yellow_background_requires_a_physical_transition(self):
        image,seed,tracks=fixture();image[:30,20:180]=(30,190,220)
        seed['geometry_reasons']=['chroma_dev 0.70 > 0.50']
        self.assertFalse(validate_yellow_illumination(image,seed,tracks)['applied'])

    def test_missing_boundary_or_other_gate_is_not_overridden(self):
        for failure in ['missing','unstable','weak','perspective','geometry','yellow']:
            image,seed,tracks=fixture();image[20:30,20:180]=(160,211,220)
            seed['geometry_reasons']=['chroma_dev 0.70 > 0.50']
            if failure=='missing':tracks['outer'][0]=None
            elif failure=='unstable':tracks['outer'][0]['resampling_stable']=False
            elif failure=='weak':tracks['outer'][0]['support']=.80
            else:seed['geometry_reasons'].append({'perspective':'fitted perspective 0.24 > 0.20','geometry':'inner frame is outside the physical card','yellow':'yellow_min 0.2 < 0.50'}[failure])
            self.assertFalse(validate_yellow_illumination(image,seed,tracks)['applied'],failure)

    def test_washout_can_be_on_any_side_without_name_or_orientation(self):
        for side in range(4):
            image,seed,tracks=self.illuminated_fixture();image[20:240,20:180]=(30,190,220);image[30:230,30:170]=(170,70,20)
            slices=[(slice(20,30),slice(20,180)),(slice(20,240),slice(170,180)),(slice(230,240),slice(20,180)),(slice(20,240),slice(20,30))]
            image[slices[side]]=(160,211,220)
            self.assertTrue(validate_yellow_illumination(image,seed,tracks)['applied'],side)

    def test_different_yellow_range_hue_is_not_lighting(self):
        image,seed,tracks=self.illuminated_fixture()
        image[20:30,20:180]=cv2.cvtColor(np.uint8([[[40,180,220]]]),cv2.COLOR_HSV2BGR)[0,0]
        result=validate_yellow_illumination(image,seed,tracks)
        self.assertFalse(result['applied']);self.assertEqual(result['reason'],'Sides disagree in hue')

    def test_border_colour_continuing_into_art_is_rejected(self):
        image,seed,tracks=self.illuminated_fixture();image[30:36,30:170]=(160,211,220)
        self.assertFalse(validate_yellow_illumination(image,seed,tracks)['applied'])

    def test_overexposure_or_darkness_loses_colour_evidence(self):
        for colour in [(250,250,250),(5,35,40)]:
            image,seed,tracks=self.illuminated_fixture();image[20:30,20:180]=colour
            self.assertFalse(validate_yellow_illumination(image,seed,tracks)['applied'])

    def test_short_observed_span_and_missing_side_are_rejected(self):
        for case in ('span','missing'):
            image,seed,tracks=self.illuminated_fixture()
            if case=='span':tracks['outer'][0]['t_range']=[.2,.8]
            else:tracks['outer'].pop()
            self.assertFalse(validate_yellow_illumination(image,seed,tracks)['applied'])

    def test_no_colour_rejection_leaves_existing_acceptance_untouched(self):
        image,seed,tracks=self.illuminated_fixture();seed['geometry_reasons']=[]
        self.assertFalse(validate_yellow_illumination(image,seed,tracks)['applied'])

    def test_original_chroma_gate_confuses_saturation_with_hue(self):
        image,seed,tracks=self.illuminated_fixture()
        outer=np.array([[19.5,19.5],[179.5,19.5],[179.5,239.5],[19.5,239.5]])
        inner=np.array([[29.5,29.5],[169.5,29.5],[169.5,229.5],[29.5,229.5]])
        original=frame_evidence(image,outer,inner)
        self.assertGreater(max(original['chroma_dev']),.50)
        self.assertGreaterEqual(min(original['yellow_in']),.85)
        result=validate_yellow_illumination(image,seed,tracks)
        self.assertTrue(result['applied'],result)
        self.assertLessEqual(np.ptp([b['median_hue'] for b in result['bands']]),8)



if __name__=='__main__':unittest.main()
