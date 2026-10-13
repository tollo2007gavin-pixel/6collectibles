import sys,cv2,numpy as np,unittest
import whitening_geometry as g
class DetectorTests(unittest.TestCase):
 def make(self,color,quad):
  image=np.full((900,900,3),205,np.uint8);q=np.float32(quad);cv2.fillConvexPoly(image,q.astype(int),color);center=q.mean(0);cv2.fillConvexPoly(image,(center+(q-center)*.84).astype(int),(190,140,90));cv2.circle(image,tuple(center.astype(int)),65,(20,40,200),-1);return image,q
 def test_blue(self):
  im,q=self.make((150,30,15),[[220,160],[620,160],[620,720],[220,720]]);result,_=g.detect(im);self.assertIsNotNone(result);self.assertLess(np.linalg.norm(q-result,axis=1).max(),8)
 def test_gold(self):
  im,q=self.make((15,130,180),[[220,160],[620,160],[620,720],[220,720]]);result,_=g.detect(im);self.assertIsNotNone(result);self.assertLess(np.linalg.norm(q-result,axis=1).max(),8)
 def test_perspective(self):
  im,q=self.make((150,30,15),[[300,130],[650,210],[570,750],[150,650]]);result,_=g.detect(im);self.assertIsNotNone(result);self.assertLess(np.linalg.norm(q-result,axis=1).max(),10)
 def test_sleeve(self):
  im,q=self.make((150,30,15),[[220,160],[620,160],[620,720],[220,720]])
  cv2.rectangle(im,(190,100),(660,780),(100,100,100),4)
  result,_=g.detect(im)
  if result is not None:self.assertLess(np.linalg.norm(q-result,axis=1).max(),10)
 def test_blank(self):self.assertIsNone(g.detect(np.zeros((600,800,3),np.uint8))[0])
if __name__=='__main__':unittest.main()
