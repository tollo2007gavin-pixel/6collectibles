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
 def test_textured_background_recovery(self):
  im,q=self.make((150,30,15),[[220,160],[620,160],[620,720],[220,720]])
  gray=np.random.default_rng(123).integers(40,160,im.shape[:2],dtype=np.uint8)
  texture=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
  mask=np.zeros(im.shape[:2],np.uint8);cv2.fillConvexPoly(mask,q.astype(int),255)
  im[mask==0]=texture[mask==0]
  result,_=g.detect(im);self.assertIsNotNone(result);self.assertLess(np.linalg.norm(q-result,axis=1).max(),g.PAD/g.CARD_W*400)
 def test_pale_blue_border(self):
  im,q=self.make((150,130,125),[[220,160],[620,160],[620,720],[220,720]])
  result,_=g.detect(im);self.assertIsNotNone(result);self.assertLess(np.linalg.norm(q-result,axis=1).max(),10)
 def test_successful_legacy_geometry_is_preserved(self):
  im,q=self.make((150,30,15),[[220,160],[620,160],[620,720],[220,720]])
  before,_=g._detect(im);after,d=g.detect(im)
  np.testing.assert_array_equal(before,after);self.assertEqual(d['recovery'],'legacy')
 def test_unknown_photo_requires_supported_blue_back(self):
  import tempfile
  from pathlib import Path
  for color,expected in [((150,30,15),True),((15,130,180),False)]:
   with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp);im,_=self.make(color,[[220,160],[620,160],[620,720],[220,720]])
    cv2.imwrite(str(folder/'original.png'),im)
    m={'photos':[{'side':'unknown','original_file':'original.png','file':'unknown.jpg'}]}
    g.refine_manifest(folder,m)
    self.assertEqual(m['has_back'],expected)
    if expected:self.assertEqual(m['photos'][0]['side'],'back')
if __name__=='__main__':unittest.main()
