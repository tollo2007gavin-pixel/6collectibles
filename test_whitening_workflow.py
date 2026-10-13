import copy
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import whitening_workflow as w

def verdict(kind='clean'):
    return {'photo_notes':'test','locations':{k:{'assessment':kind,'severity':0 if kind=='clean' else 1 if kind=='whitening' else None,'confidence':.9,'reason':'visible printed border evidence','photo_ids':['back_1']} for k in w.keys(['back'])}}

def packet():return {'images':[b'test'],'descriptions':['back_1'],'photos':[{'photo_id':'back_1'}],'warnings':[],'missing_sides':[]}

class WorkflowTests(unittest.TestCase):
    def test_experimental_entry_point_is_explicit(self):
        import whitening_scan as old
        with patch.object(sys,'argv',['whitening_scan.py']),patch.object(old,'legacy_main',return_value=0) as legacy:
            self.assertEqual(old.main(),0);legacy.assert_called_once()
        with patch.object(sys,'argv',['whitening_scan.py','--triage-v2','--cards','example']),patch.object(w,'main',return_value=0) as experimental:
            self.assertEqual(old.main(),0);experimental.assert_called_once_with(['--cards','example'])
    def test_cli_cache_invalidation_and_evidence_output(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);card=root/'cards'/'abc';card.mkdir(parents=True)
            (card/'manifest.json').write_text(json.dumps({'card_id':'abc','photos':[]}))
            out=root/'results';p=packet()
            with patch.object(w,'prepare',return_value=p),patch.object(w.providers,'load_dotenv'),patch.object(w.providers,'call_openai',return_value=(json.dumps(verdict()),100,100)) as call:
                args=['--cards',str(root/'cards'),'--out',str(out)]
                self.assertEqual(w.main(args),0);self.assertEqual(call.call_count,3)
                w.main(args);self.assertEqual(call.call_count,3)
                p['images']=[b'changed photo'];w.main(args);self.assertEqual(call.call_count,6)
                w.main(args+['--min-confidence','.8']);self.assertEqual(call.call_count,9)
            r=json.loads((out/'abc.json').read_text());self.assertEqual(r['decision'],'clean');self.assertEqual(len(r['primary']),3)
            self.assertTrue((out/'shortlist_v2.csv').exists());self.assertFalse((card/'whitening.json').exists())

    def test_cli_invalid_json_retry_and_review_queue(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);card=root/'cards'/'abc';card.mkdir(parents=True)
            (card/'manifest.json').write_text(json.dumps({'card_id':'abc','photos':[]}))
            out=root/'results'
            with patch.object(w,'prepare',return_value=packet()),patch.object(w.providers,'load_dotenv'),patch.object(w.providers,'call_openai',return_value=('bad JSON',100,100)) as call:
                args=['--cards',str(root/'cards'),'--out',str(out)]
                w.main(args);w.main(args);self.assertEqual(call.call_count,6)
            r=json.loads((out/'abc.json').read_text());self.assertEqual(r['status'],'failed');self.assertEqual(r['decision'],'needs_review')
            self.assertIn('abc',(out/'needs_review.csv').read_text())

    def test_valid_verdicts(self):
        for k in ['clean','whitening','uncertain']:self.assertTrue(w.validate(verdict(k),['back'],{'back_1'}))
    def test_invalid_evidence(self):
        for field,value in [('confidence',float('nan')),('confidence',True),('severity',True),('reason',''),('photo_ids',['fake']),('photo_ids',[]),('assessment','probably')]:
            v=verdict();v['locations']['back.TL'][field]=value
            with self.subTest(field=field):self.assertFalse(w.validate(v,['back'],{'back_1'}))
    def test_inconsistent_severity(self):
        v=verdict('whitening');v['locations']['back.TL']['severity']=0
        self.assertFalse(w.validate(v,['back'],{'back_1'}))
    def test_missing_location(self):
        v=verdict();del v['locations']['back.TL'];self.assertFalse(w.validate(v,['back'],{'back_1'}))
    def test_wrong_side_citation(self):
        v=verdict();v['locations']['back.TL']['photo_ids']=['front_1'];self.assertFalse(w.validate(v,['back'],{'front_1','back_1'}))
    def test_unanimous_clean(self):
        self.assertEqual(w.resolve([verdict()]*3,[],['back'],.6,3,2)[0],'clean')
    def test_zero_slips_no_unreviewed_override(self):
        self.assertEqual(w.resolve([verdict(),verdict(),verdict('whitening')],[],['back'],.6,3,2)[0],'needs_review')
    def test_independent_artifact_resolution(self):
        self.assertEqual(w.resolve([verdict('whitening')]*3,[verdict()]*2,['back'],.6,3,2)[0],'clean')
    def test_confirmed_whitening(self):
        self.assertEqual(w.resolve([verdict('whitening')]*3,[verdict('whitening')]*2,['back'],.6,3,2)[0],'whitening_detected')
    def test_review_disagreement(self):
        self.assertEqual(w.resolve([verdict('whitening')]*3,[verdict(),verdict('whitening')],['back'],.6,3,2)[0],'needs_review')
    def test_uncertain_is_not_damage(self):
        self.assertEqual(w.resolve([verdict('uncertain')]*3,[verdict('uncertain')]*2,['back'],.6,3,2)[0],'needs_review')
    def test_low_confidence_is_review(self):
        v=verdict();v['locations']['back.TL']['confidence']=.1
        self.assertEqual(w.resolve([v]*3,[v]*2,['back'],.6,3,2)[0],'needs_review')
    def test_failed_primary_cannot_be_rescued(self):
        self.assertEqual(w.resolve([verdict()],[verdict()]*2,['back'],.6,3,2)[0],'needs_review')
    def test_one_review_insufficient(self):
        self.assertEqual(w.resolve([verdict('whitening')]*3,[verdict()],['back'],.6,3,2)[0],'needs_review')
    def test_missing_photos_no_api(self):
        p=packet();p['missing_sides']=['back']
        r=w.scan_card(Path('.'),{'card_id':'test'},p,caller=lambda *a:self.fail('API called'))
        self.assertEqual(r['decision'],'needs_review')
    def test_disputed_locations_in_review_prompt(self):
        prompts=[]
        def caller(model,prompt,images,sides):
            prompts.append(prompt)
            v=verdict('whitening') if len(prompts)<=3 else verdict()
            return {'verdict':v,'estimated_cost_usd':.001,'error':None}
        r=w.scan_card(Path('.'),{'card_id':'test'},packet(),caller=caller)
        self.assertEqual(len(prompts),5);self.assertIn('back.TL',prompts[3]);self.assertEqual(r['decision'],'clean')
    def test_failed_review_safe_and_cost_unknown(self):
        def caller(*a):raise RuntimeError('network unavailable')
        r=w.scan_card(Path('.'),{'card_id':'test'},packet(),caller=caller)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['decision'],'needs_review');self.assertIsNone(r['estimated_cost_usd'])
    def test_fingerprint_changes_with_settings_and_images(self):
        p=packet();a=w.fingerprint(p,{'model':'a'});self.assertNotEqual(a,w.fingerprint(p,{'model':'b'}))
        p['images']=[b'new'];self.assertNotEqual(a,w.fingerprint(p,{'model':'a'}))
    def test_queues_keep_uncertain_cards(self):
        with tempfile.TemporaryDirectory() as t:
            w.write_queues(Path(t),[{'card_id':'abc','decision':'needs_review','reason':'glare'}])
            self.assertIn('abc',(Path(t)/'needs_review.csv').read_text());self.assertNotIn('abc',(Path(t)/'shortlist_v2.csv').read_text())
    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):w.local_file(Path(t),'../secret')

if __name__=='__main__':unittest.main(verbosity=2)
