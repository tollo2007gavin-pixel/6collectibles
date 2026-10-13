"""Pixel-evidence regression checks for desaturated border validation."""
import copy
import unittest
import numpy as np
from yellow_batch_engine.pale_border import validate_pale_border


def fixture():
    image=np.full((260,200,3),20,np.uint8)
    image[20:240,20:180]=(30,190,220)
    image[30:230,30:170]=(170,70,20)
    image[20:30,20:180]=(175,183,183)
    quads={'outer':np.array([[19.5,19.5],[179.5,19.5],[179.5,239.5],[19.5,239.5]]),
           'inner':np.array([[29.5,29.5],[169.5,29.5],[169.5,229.5],[29.5,229.5]])}
    tracks={}
    for kind,q in quads.items():
        tracks[kind]=[]
        for i in range(4):
            t=np.linspace(.035,.965,160)[:,None]
            tracks[kind].append(dict(points=(q[i]+t*(q[(i+1)%4]-q[i])).tolist(),t_range=[.035,.965],
                                     retained=True,support=1.,resampling_stable=True,covered_segments=12))
    seed=dict(ppm=160/63,geometry_reasons=['yellow_min 0.03 < 0.50','chroma_dev 0.84 > 0.50'])
    return image,seed,tracks


class PaleBorderTests(unittest.TestCase):
    def test_pale_yellow_with_two_real_transitions_is_recovered_without_mutation(self):
        image,seed,tracks=fixture();original=copy.deepcopy((seed,tracks))
        evidence=validate_pale_border(image,seed,tracks)
        self.assertTrue(evidence['applied'],evidence)
        self.assertEqual(evidence['side'],0)
        self.assertEqual((seed,tracks),original)

    def test_neutral_holder_strip_does_not_qualify_as_pale_yellow(self):
        image,seed,tracks=fixture();image[20:30,20:180]=183
        self.assertFalse(validate_pale_border(image,seed,tracks)['applied'])

    def test_coloured_artwork_is_not_accepted_as_desaturated_yellow(self):
        image,seed,tracks=fixture();image[20:30,20:180]=(183,175,183)
        self.assertFalse(validate_pale_border(image,seed,tracks)['applied'])

    def test_yellow_continuing_into_art_has_no_valid_interior_transition(self):
        image,seed,tracks=fixture();image[30:35,30:170]=(175,183,183)
        self.assertFalse(validate_pale_border(image,seed,tracks)['applied'])

    def test_missing_or_unstable_track_cannot_use_colour_recovery(self):
        for change in ['missing','unstable','weak']:
            image,seed,tracks=fixture()
            if change=='missing':tracks['inner'][3]=None
            elif change=='unstable':tracks['outer'][1]['resampling_stable']=False
            else:tracks['inner'][2]['support']=.80
            self.assertFalse(validate_pale_border(image,seed,tracks)['applied'],change)

    def test_two_pale_sides_need_other_evidence(self):
        image,seed,tracks=fixture();image[20:240,170:180]=(175,183,183)
        self.assertFalse(validate_pale_border(image,seed,tracks)['applied'])

    def test_other_rejection_reasons_cannot_be_overridden(self):
        for reason in ['fitted perspective 0.193 > 0.100','inner frame is outside the physical card',
                       'yellow border continues into crop at sides: bottom']:
            image,seed,tracks=fixture();seed['geometry_reasons'].append(reason)
            self.assertFalse(validate_pale_border(image,seed,tracks)['applied'])


if __name__=='__main__':unittest.main()
