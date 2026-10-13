from pathlib import Path
import tempfile,unittest,json
from unittest.mock import patch
import cv2,numpy as np
import whitening_inputs as w
import organize_listings as org
class InputsTests(unittest.TestCase):
 def test_invalid_and_off_frame_boundaries(self):
  self.assertFalse(w.geometry(None,(400,400,3))[0])
  self.assertFalse(w.geometry(np.array([[0,0],[200,0],[200,280],[0,280]]),(400,400,3))[0])
  self.assertTrue(w.geometry(np.array([[40,40],[240,40],[240,320],[40,320]]),(400,400,3))[0])
 def test_local_path_escape(self):
  with self.assertRaises(ValueError):w.local(Path('cards'), '../secret')
 def test_focus_prefers_clear_pixels(self):
  a=np.zeros((200,200,3),np.uint8);a[:,::4]=255;b=cv2.GaussianBlur(a,(21,21),5)
  self.assertGreater(w.region_quality(a,[0,0,200,200])['score'],w.region_quality(b,[0,0,200,200])['score'])
 def candidate(self,pid,score,pixel='a'):
  return {'photo_id':pid,'valid_boundary':True,'score':score,'pixel_sha256':pixel,'quality':{k:{'score':score} for k in w.LOCS}}
 def test_duplicate_decoded_pixels_not_another_view(self):
  chosen,_=w.choose([self.candidate('a',.8),self.candidate('b',.7)])
  self.assertEqual(len(chosen),1)
 def test_second_view_covers_weak_location(self):
  a=self.candidate('a',.8);a['quality']['TL']['score']=.1;b=self.candidate('b',.6,'b')
  chosen,alternatives=w.choose([a,b]);self.assertEqual(len(chosen),2);self.assertIn('TL',alternatives[0]['improved_locations'])
 def test_native_crop_pixel_identity_and_original_bytes(self):
  rng=np.random.default_rng(2);img=rng.integers(0,256,(400,400,3),dtype=np.uint8);quad=np.array([[40,40],[240,40],[240,320],[40,320]],np.float32)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=root/'source.png';src.write_bytes(w.png(img))
   with patch.object(w,'find_card_quad',return_value=quad):
    # analyze default is explicitly bound, use a patched wrapper.
    original=w.analyze
    with patch.object(w,'analyze',side_effect=lambda p,pid,m:original(p,pid,m,detector=lambda _:quad)):
     rec=w.prepare(root,{'card_id':'a','photos':[{'side':'back','original_file':'source.png'}]},root/'packet')
   a=rec['assets'][0];self.assertEqual((root/'packet'/a['context']).read_bytes(),src.read_bytes())
   for loc,box in a['source_boxes'].items():
    x1,y1,x2,y2=box;crop=cv2.imread(str(root/'packet'/f'{a["photo_id"]}_{loc}.png'))
    self.assertTrue(np.array_equal(crop,img[y1:y2,x1:x2]))
 def test_organizer_preserves_downloaded_bytes(self):
  img=np.zeros((100,100,3),np.uint8);raw=cv2.imencode('.jpg',img)[1].tobytes()
  with tempfile.TemporaryDirectory() as tmp,patch.object(org,'find_card',return_value=(img,False)):
   m=org.process_card('a',[('https://example/image',img,raw)],Path(tmp),{})
   self.assertEqual((Path(tmp)/'a'/m['photos'][0]['original_file']).read_bytes(),raw)
 def test_organizer_fallback_is_lossless(self):
  img=np.random.default_rng(3).integers(0,256,(100,100,3),dtype=np.uint8)
  with tempfile.TemporaryDirectory() as tmp,patch.object(org,'find_card',return_value=(img,False)):
   m=org.process_card('a',[('x',img)],Path(tmp),{});p=Path(tmp)/'a'/m['photos'][0]['original_file']
   self.assertEqual(p.suffix,'.png');self.assertTrue(np.array_equal(cv2.imread(str(p)),img))
 def packet(self):
  return {'status':'ready','assets':[{'role':'initial','photo_id':'p1','context':'c1','sheet':'s1'},{'role':'reserve','photo_id':'p2','context':'c2','sheet':'s2'}]}
 def verdict(self,pid,assessment='clean'):
  return {k:{'assessment':assessment,'evidence':'visible evidence','photo_id':pid} for k in w.LOCS}
 def test_clear_result_needs_one_call(self):
  with patch('builtins.print'),patch.object(w,'validate_response',wraps=w.validate_response):
   calls=[]
   def caller(*args):calls.append(args);return self.verdict('p1')
   r=w.assess(self.packet(),caller);self.assertEqual(r['decision'],'clean');self.assertEqual(len(calls),1)
 def test_uncertainty_uses_another_source(self):
  first=self.verdict('p1');first['TL']['assessment']='uncertain';calls=[]
  def caller(files,*_):calls.append(files);return first if len(calls)==1 else self.verdict('p2')
  r=w.assess(self.packet(),caller);self.assertEqual(r['calls'],2);self.assertEqual(calls[1],['c2','p2_TL.png']);self.assertEqual(r['decision'],'clean')
 def test_any_whitening_fails_zero_slip(self):
  first=self.verdict('p1');first['BR']['assessment']='whitening'
  r=w.assess(self.packet(),lambda *_:first);self.assertEqual(r['decision'],'not_clean');self.assertEqual(r['calls'],1)
 def test_invalid_or_missing_evidence_cannot_clear(self):
  self.assertEqual(w.assess(self.packet(),lambda *_:{})['decision'],'needs_review')
  self.assertEqual(w.assess({'status':'needs_review','reason':'bad_crop'},lambda *_:self.fail('Must not call'))['calls'],0)
 def test_uncertain_other_view_cannot_fill_by_corner_name(self):
  first=self.verdict('p1');first['TL']['assessment']='uncertain';second=self.verdict('p2');second['BR']['assessment']='uncertain';answers=iter([first,second])
  self.assertEqual(w.assess(self.packet(),lambda *_:next(answers))['decision'],'needs_review')
if __name__=='__main__':unittest.main()
