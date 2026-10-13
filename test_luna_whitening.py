import sys,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
import luna_whitening as l
class Tests(unittest.TestCase):
 def setUp(self):
  import tempfile,hashlib
  import cv2,numpy as np
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.im=Path(self.temp.name)/'synthetic.png';cv2.imwrite(str(self.im),np.zeros((10,10,3),np.uint8))
  digest=hashlib.sha256(self.im.read_bytes()).hexdigest()
  self.bank=[{'card_id':str(i),'label':'clean' if i<8 else 'not_clean','path':str(self.im),'sha256':digest} for i in range(16)]
  self.refs=l.references(self.bank,'0')
 def run_decisions(self,ds):
  calls=[]
  def caller(body):
   calls.append(body);return {'decision':ds[len(calls)-1],'location':'edge','evidence':'visible evidence','cost_usd':.01}
  r=l.classify(self.im,self.im,self.im,self.refs,caller);return r,calls
 def test_all_target_exclusions(self):
  for row in self.bank:
   rr=l.references(self.bank,row['card_id']);self.assertEqual(len(rr),8);self.assertNotIn(row['card_id'],[x['card_id'] for x in rr]);self.assertEqual(sum(x['label']=='clean' for x in rr),4)
 def test_unseen(self):self.assertEqual(len(l.references(self.bank,'unseen')),8)
 def test_clean_agreement(self):
  r,c=self.run_decisions(['clean','clean']);self.assertEqual(r['decision'],'clean');self.assertEqual(len(c),2)
 def test_damage_agreement(self):
  r,c=self.run_decisions(['not_clean','not_clean']);self.assertEqual(r['decision'],'not_clean');self.assertEqual(len(c),2)
 def test_disagreement(self):
  r,c=self.run_decisions(['clean','not_clean','clean']);self.assertEqual(r['decision'],'clean');self.assertEqual(len(c),3);self.assertAlmostEqual(r['cost_usd'],.03)
 def test_uncertain(self):
  r,c=self.run_decisions(['needs_review','needs_review']);self.assertEqual(r['decision'],'needs_review')
 def test_invalid_stops(self):
  r=l.classify(self.im,self.im,self.im,self.refs,lambda b:{'decision':'needs_review','invalid':True,'cost_usd':.02});self.assertEqual(len(r['stages']),1);self.assertEqual(r['cost_usd'],.02)
 def test_no_identifiers(self):
  b=l.request(self.im,self.im,None,self.refs);text=json.dumps(b);self.assertNotIn('117386886277',text);self.assertNotIn('CLEAN_',text);self.assertNotIn('score1_',text)
 def test_portable_bank_does_not_mutate_paths(self):
  before=json.dumps(self.bank);refs=l.references(self.bank,'unseen')
  self.assertEqual(json.dumps(self.bank),before)
  self.assertTrue(all(Path(r['path']).is_absolute() for r in refs))
if __name__=='__main__':unittest.main()
