import unittest
import numpy as np
from pixel_whitening import grain_measurement

class GrainTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(7)
        residual=rng.normal(0,7,(500,500)).astype(np.float32)
        component=np.zeros((500,500),bool);component[249:252,249:252]=True
        residual[component]=17
        return component,residual,np.zeros_like(residual),np.ones_like(component)

    def test_weak_grain_rejected_but_strong_chip_preserved(self):
        c,r,s,b=self.fixture()
        self.assertTrue(grain_measurement(c,r,s,b)['exclude'])
        r[c]=45
        self.assertFalse(grain_measurement(c,r,s,b)['exclude'])
        r[c]=17;s[c]=45
        self.assertFalse(grain_measurement(c,r,s,b)['exclude'])

    def test_large_and_unreliable_reference_preserved(self):
        c,r,s,b=self.fixture();c[245:255,245:255]=True;r[c]=17
        self.assertFalse(grain_measurement(c,r,s,b)['exclude'])
        b[:]=False
        self.assertEqual(grain_measurement(c,r,s,b)['status'],'insufficient_reference')
        b[:]=True;r*=5
        self.assertEqual(grain_measurement(c,r,s,b)['status'],'unstable_reference')

if __name__=='__main__':unittest.main()
