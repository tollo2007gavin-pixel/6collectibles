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
 def test_bundled_reference_bank(self):
  bank=json.loads((ROOT/'luna_references.json').read_text())
  self.assertEqual(len(bank),16)
  self.assertEqual(len({r['card_id'] for r in bank}),16)
  for target in ['unseen']+[r['card_id'] for r in bank]:
   refs=l.references(bank,target)
   self.assertEqual(len(refs),8)
   self.assertEqual(sum(r['label']=='clean' for r in refs),4)
   self.assertNotIn(target,[r['card_id'] for r in refs])
 def run_decisions(self,ds):
  calls=[]
  def caller(body):
   calls.append(body);return {'decision':ds[len(calls)-1],'location':'edge','evidence':'visible evidence','cost_usd':.01}
  r=l.classify(self.im,self.im,self.im,self.refs,caller,parallel=False);return r,calls
 def test_provenance_records_inputs_without_paths(self):
  import hashlib
  r,_=self.run_decisions(['clean','clean']);p=r['provenance']
  self.assertEqual(p['prompt_sha256'],hashlib.sha256(l.SYSTEM.encode()).hexdigest())
  self.assertEqual(p['input_sha256']['context'],hashlib.sha256(self.im.read_bytes()).hexdigest())
  self.assertNotIn(str(self.im),json.dumps(p))
  self.assertEqual(len(p['references']),8)
 def test_missing_input_prevents_api_call(self):
  calls=[]
  r=l.classify(self.im.with_name('missing.png'),self.im,self.im,self.refs,lambda b:calls.append(b))
  self.assertEqual(calls,[]);self.assertEqual(r['decision'],'needs_review')
 def test_parallel_overlap_and_stable_order(self):
  from threading import Barrier
  barrier=Barrier(2)
  def caller(body):
   has_tips=any(x.get('text','').startswith('Enlarged corner') for x in body['messages'][1]['content'])
   barrier.wait(timeout=3)
   return {'decision':'clean','location':'tips' if has_tips else 'standard','evidence':'ok','cost_usd':.01}
  r=l.classify(self.im,self.im,self.im,self.refs,caller)
  self.assertNotIn('error',r)
  self.assertEqual([s['location'] for s in r['stages']],['standard','tips'])
  self.assertEqual(len(r['timing']['attempts']),2)
  self.assertGreater(r['timing']['total_seconds'],0)
 def test_parallel_error_preserves_other_result_and_cost(self):
  def caller(body):
   has_tips=any(x.get('text','').startswith('Enlarged corner') for x in body['messages'][1]['content'])
   if has_tips:raise RuntimeError('simulated failure')
   return {'decision':'clean','location':'edge','evidence':'ok','cost_usd':.01}
  r=l.classify(self.im,self.im,self.im,self.refs,caller)
  self.assertEqual(r['decision'],'needs_review');self.assertEqual(len(r['stages']),1)
  self.assertEqual(len(r['timing']['attempts']),2);self.assertEqual(r['cost_usd'],.01)
 def test_parallel_disagreement_waits_for_both(self):
  from threading import Barrier
  barrier=Barrier(2)
  def caller(body):
   texts=[x.get('text','') for x in body['messages'][1]['content']]
   judge=any(x.startswith('Two fallible') for x in texts)
   if not judge:barrier.wait(timeout=3)
   tips=any(x.startswith('Enlarged corner') for x in texts)
   return {'decision':'not_clean' if tips else 'clean','location':'edge','evidence':'ok','cost_usd':.01}
  r=l.classify(self.im,self.im,self.im,self.refs,caller)
  self.assertEqual(r['decision'],'not_clean');self.assertEqual(len(r['stages']),3)
  self.assertEqual(r['timing']['attempts'][-1]['inspection'],'tie_breaker')
 def test_parallel_invalid_records_both_calls_without_judge(self):
  r=l.classify(self.im,self.im,self.im,self.refs,lambda body:{'decision':'needs_review','invalid':True,'cost_usd':.02})
  self.assertEqual(r['decision'],'needs_review');self.assertEqual(len(r['stages']),2)
  self.assertAlmostEqual(r['cost_usd'],.04)
 def test_shared_concurrency_cap(self):
  import time
  from threading import Lock
  from concurrent.futures import ThreadPoolExecutor
  lock=Lock();active=0;peak=0
  def caller(body):
   nonlocal active,peak
   with lock:active+=1;peak=max(peak,active)
   time.sleep(.02)
   with lock:active-=1
   return {'decision':'clean','location':'','evidence':'','cost_usd':0}
  with ThreadPoolExecutor(max_workers=4) as pool:
   results=list(pool.map(lambda _:l.classify(self.im,self.im,self.im,self.refs,caller),range(4)))
  self.assertLessEqual(peak,3);self.assertGreater(peak,1)
  self.assertTrue(all(r['decision']=='clean' for r in results))
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
  r=l.classify(self.im,self.im,self.im,self.refs,lambda b:{'decision':'needs_review','invalid':True,'cost_usd':.02},parallel=False);self.assertEqual(len(r['stages']),1);self.assertEqual(r['cost_usd'],.02)
 def test_no_identifiers(self):
  b=l.request(self.im,self.im,None,self.refs);text=json.dumps(b);self.assertNotIn('117386886277',text);self.assertNotIn('CLEAN_',text);self.assertNotIn('score1_',text)
 def test_portable_bank_does_not_mutate_paths(self):
  before=json.dumps(self.bank);refs=l.references(self.bank,'unseen')
  self.assertEqual(json.dumps(self.bank),before)
  self.assertTrue(all(Path(r['path']).is_absolute() for r in refs))
if __name__=='__main__':unittest.main()
