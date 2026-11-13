import unittest
import cv2
import numpy as np
from interior_art import estimate

class InteriorArtTests(unittest.TestCase):
    def fixture(self,size=500):
        image=np.full((size,size,3),(140,45,20),np.uint8)
        cv2.rectangle(image,(0,0),(round(size*.2),round(size*.18)),(180,180,180),-1)
        inside=np.ones((size,size),bool)
        zone=np.zeros_like(inside);zone[:,round(size*.8):]=True;zone[round(size*.8):]=True
        return image,inside,zone

    def test_all_corners_and_sizes(self):
        for size in [250,500,1000]:
            for corner,flip in {'BR':lambda a:a,'BL':np.fliplr,'TR':np.flipud,
                                'TL':lambda a:np.flipud(np.fliplr(a))}.items():
                im,inside,zone=self.fixture(size)
                info,mask=estimate(flip(im).copy(),corner,flip(inside),flip(zone),'agreement')
                self.assertEqual(info['status'],'detected')
                normal=flip(mask)
                self.assertTrue(normal[round(size*.1),round(size*.1)])
                self.assertFalse(normal[round(size*.5),round(size*.5)])
                self.assertFalse(np.any(mask & flip(zone)))

    def test_detached_light_spot_and_small_patch_not_art(self):
        for box in [(200,200,280,280),(0,0,12,12)]:
            im,inside,zone=self.fixture();im[:]=(140,45,20)
            cv2.rectangle(im,box[:2],box[2:],(220,220,220),-1)
            info,mask=estimate(im,'BR',inside,zone,'agreement')
            self.assertFalse(mask.any())

    def test_uncertain_edges_and_connected_edge_damage_not_excluded(self):
        im,inside,zone=self.fixture()
        self.assertFalse(estimate(im,'BR',inside,zone,'unconfirmed')[1].any())
        zone[:110,:110]=True
        self.assertFalse(estimate(im,'BR',inside,zone,'agreement')[1].any())
        im,inside,zone=self.fixture();im[:]=(140,45,20)
        cv2.rectangle(im,(420,300),(499,499),(220,220,220),-1)
        self.assertFalse(estimate(im,'BR',inside,zone,'agreement')[1].any())

    def test_noisy_gray_art_and_mask_stays_off_edge(self):
        im,inside,zone=self.fixture()
        rng=np.random.default_rng(11)
        im=np.clip(im.astype(float)+rng.normal(0,10,im.shape),0,255).astype(np.uint8)
        _,mask=estimate(im,'BR',inside,zone,'agreement')
        self.assertTrue(mask[40,40])
        self.assertFalse((mask&zone).any())

    def test_clipped_art_strip_in_all_orientations_and_scales(self):
        for size in [250,500,1000]:
            for corner,flip in {'BR':lambda a:a,'BL':np.fliplr,'TR':np.flipud,
                                'TL':lambda a:np.flipud(np.fliplr(a))}.items():
                im,inside,zone=self.fixture(size);im[:]=(140,45,20)
                im[:round(size*.268),:round(size*.048)]=(180,180,180)
                info,mask=estimate(flip(im).copy(),corner,flip(inside),flip(zone),'agreement')
                self.assertEqual(info['rectangles'][0]['layout'],'clipped_strip')
                self.assertTrue(flip(mask)[round(size*.24),round(size*.02)])
                self.assertFalse((mask&flip(zone)).any())

    def test_small_thin_or_detached_strips_are_not_art(self):
        for box in [(0,0,5,134),(0,0,24,50),(20,20,44,154)]:
            im,inside,zone=self.fixture();im[:]=(140,45,20)
            cv2.rectangle(im,box[:2],box[2:],(180,180,180),-1)
            self.assertFalse(estimate(im,'BR',inside,zone,'agreement')[1].any())

if __name__=='__main__':unittest.main()
