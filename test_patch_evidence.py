import unittest
import cv2
import numpy as np
from patch_evidence import prepare,measure


class PatchEvidenceTests(unittest.TestCase):
    def test_connected_patch_not_just_threshold_core_and_normalized_scales(self):
        areas=[]
        for size in (250,500,1000):
            im=np.full((size,size,3),(140,45,20),np.uint8)
            center=(round(size*.5),round(size*.5))
            cv2.circle(im,center,round(size*.018),(190,190,190),-1)
            core=np.zeros((size,size),bool)
            x,y=center;core[y:y+max(1,round(size*.004)),x:x+max(1,round(size*.004))]=True
            rows=measure(core,prepare(im),{},np.zeros_like(core))['scales']
            self.assertTrue(all(r['status']=='measured' for r in rows))
            self.assertTrue(all(r['area_at_500px']>40 for r in rows))
            self.assertTrue(all(r['axis_ratio']<1.2 for r in rows))
            areas.append(rows[-1]['area_at_500px'])
        self.assertLess(max(areas)/min(areas),1.5)

    def test_light_line_is_clipped_or_elongated(self):
        im=np.full((500,500,3),(140,45,20),np.uint8)
        cv2.line(im,(0,250),(499,250),(210,210,210),4)
        core=np.zeros((500,500),bool);core[248:253,248:253]=True
        rows=measure(core,prepare(im),{},np.zeros_like(core))['scales']
        self.assertTrue(all(r['status']=='clipped' or r.get('axis_ratio',0)>3 for r in rows))

    def test_far_bright_object_does_not_attach_to_core(self):
        im=np.full((500,500,3),(140,45,20),np.uint8)
        cv2.circle(im,(280,250),12,(230,230,230),-1)
        core=np.zeros((500,500),bool);core[250:252,250:252]=True
        rows=measure(core,prepare(im),{},np.zeros_like(core))['scales']
        self.assertTrue(all(r['status']!='measured' for r in rows))


if __name__=='__main__':unittest.main()
