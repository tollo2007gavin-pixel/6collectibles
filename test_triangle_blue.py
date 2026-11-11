import unittest
import numpy as np
from pixel_whitening import triangle_blue

class TriangleTests(unittest.TestCase):
    def fixture(self):
        component=np.zeros((100,100),bool)
        component[50,50]=True
        hsv=np.full((100,100,3),(110,200,100),np.uint8)
        return component,hsv

    def test_equilateral_and_all_four_directions(self):
        for corner,signs in [('TR',(1,-1)),('TL',(-1,-1)),('BL',(-1,1)),('BR',(1,1))]:
            component,hsv=self.fixture()
            result=triangle_blue(component,hsv,corner,.1)
            points=np.array(result['vertices'])
            np.testing.assert_allclose(points[0],[50,50])
            lengths=[np.linalg.norm(points[i]-points[(i+1)%3]) for i in range(3)]
            np.testing.assert_allclose(lengths,[20/np.sqrt(3)]*3)
            delta=np.array(result['base_center'])-points[0]
            np.testing.assert_equal(np.sign(delta),signs)
            self.assertAlmostEqual(np.linalg.norm(delta),10)
            self.assertTrue(result['complete_triangle'])
            self.assertEqual(result['blue_fraction'],1)

    def test_background_and_cropped_triangle(self):
        component,hsv=self.fixture()
        hsv[:50,51:] = (0,0,200)
        self.assertLess(triangle_blue(component,hsv,'TR',.1)['blue_fraction'],.1)
        self.assertEqual(triangle_blue(component,hsv,'BL',.1)['blue_fraction'],1)
        component[:]=False; component[1,98]=True
        self.assertFalse(triangle_blue(component,hsv,'TR',.1)['complete_triangle'])

if __name__=='__main__': unittest.main()
