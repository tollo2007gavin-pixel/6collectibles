import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import contextlib
import io
from unittest.mock import patch, Mock

sys.path.insert(0, os.environ.get('AUDIT_SOURCE', str(Path(__file__).resolve().parent)))
import ebay_fetch as ef
import organize_listings as org
import providers as pr
import whitening_scan as ws
import numpy as np

def verdict():
    return {'photo_usable': True, 'corners': dict.fromkeys(['TL','TR','BL','BR'], 0),
            'edges': dict.fromkeys(['TOP','BOTTOM','LEFT','RIGHT'], 0),
            'overall': 0, 'confidence': .9, 'notes': 'test'}

def item(i):
    return {'itemId': i, 'title': 'pokemon charizard', 'image': {'imageUrl': 'https://i.ebayimg.com/images/g/test/s-l500.jpg'}}

def response(data):
    r = Mock(ok=True)
    r.json.return_value = data
    return r

class RegressionTests(unittest.TestCase):
    def test_failed_image_write_does_not_mark_listing_seen(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cfg=root/'config.yaml';cfg.write_text('searches: [{q: test}]\ndownload_workers: 1\n')
            listing=ef.Listing('v1|123|0','test','1','USD','https://www.ebay.com/itm/123','https://i.ebayimg.com/images/g/abc/s-l500.jpg','test')
            argv=['ebay_fetch.py','--config',str(cfg),'--state',str(root/'seen.json'),'--out',str(root/'out')]
            with patch.object(sys,'argv',argv),patch.dict(os.environ,{'EBAY_CLIENT_ID':'fake','EBAY_CLIENT_SECRET':'fake'}), \
                 patch.object(ef,'get_app_token',return_value='fake'),patch.object(ef,'search_listings',return_value=([listing],1)), \
                 patch.object(ef,'fetch_image',return_value=(listing,np.zeros((10,10,3),np.uint8),'ok',True)),patch.object(ef.cv2,'imwrite',return_value=False):
                ef.main()
            state=json.loads((root/'seen.json').read_text())
            self.assertNotIn(listing.item_id,state['items'])
            result=json.loads((root/'out'/'results.jsonl').read_text())
            self.assertEqual(result['status'],'error')

    def scan_fixture(self, root, caller, argv_extra=(), previous=None):
        d=root/'card'; d.mkdir(exist_ok=True)
        (d/'manifest.json').write_text(json.dumps({'card_id':'card','photos':[{'file':'back.jpg','side':'back','card_found':True,'sharpness':100}]}))
        if previous is not None: (d/'whitening.json').write_text(json.dumps(previous))
        with patch.object(sys,'argv',['whitening_scan.py','--cards',str(root),*argv_extra]), \
             patch.dict(ws.CALLERS,{'openai':caller}), patch.object(ws,'build_sheets',return_value=([b'fake'],None)):
            ws.main()
        return json.loads((d/'whitening.json').read_text()) if (d/'whitening.json').exists() else None

    def test_partial_calls_not_permanently_cached(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            caller=Mock(side_effect=[(json.dumps(verdict()),1,1),pr.APIError('temporary'),pr.APIError('temporary')])
            result=self.scan_fixture(root,caller)
            self.assertEqual(result['status'],'failed')
            retry=Mock(return_value=(json.dumps(verdict()),1,1))
            result=self.scan_fixture(root,retry)
            self.assertEqual(retry.call_count,3)
            self.assertTrue(result['clean'])

    def test_failed_rescan_removes_stale_shortlist_entry(self):
        with tempfile.TemporaryDirectory() as t:
            result=self.scan_fixture(Path(t),Mock(side_effect=pr.APIError('temporary')),['--rescan'],
                       {'card_id':'card','status':'ok','clean':True})
            self.assertFalse(result.get('clean'))
            self.assertEqual(result['status'],'failed')
            self.assertNotIn('card,',(Path(t)/'shortlist.csv').read_text())

    def test_confirmation_failure_is_retryable(self):
        with tempfile.TemporaryDirectory() as t:
            caller=Mock(side_effect=[(json.dumps(verdict()),1,1)]*3+[pr.APIError('temporary')])
            result=self.scan_fixture(Path(t),caller,['--confirm-model','strong'])
            self.assertEqual(result['status'],'failed')
            self.assertFalse(result['clean'])

    def test_refusal_and_truncation_cannot_parse(self):
        for reason, refusal in [('length',None),('stop','cannot inspect')]:
            payload={'choices':[{'finish_reason':reason,'message':{'content':json.dumps(verdict()),'refusal':refusal}}]}
            with patch.dict(os.environ,{'OPENAI_API_KEY':'fake'}),patch.object(pr,'_post',return_value=payload):
                result,_,_=pr.call_openai('test','test','test',[])
            self.assertIsNone(ws.parse_json(result))

    def test_invalid_schema_is_not_silently_downgraded(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'fake'}),patch.object(pr,'_post',side_effect=pr.APIError('400: invalid json_schema required keys')) as post:
            with self.assertRaises(pr.APIError): pr.call_openai('schema-test','test','test',[])
            self.assertEqual(post.call_count,1)

    def test_unknown_location_stops_early(self):
        v=verdict();v['edges']['TOP']=None
        self.assertTrue(ws._already_failed([v],1))

    def test_complete_and_incomplete_manifest_resume(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'photo.jpg').write_bytes(b'test')
            p=root/'manifest.json'
            p.write_text(json.dumps({'photos':[{'file':'photo.jpg'}]}))
            self.assertTrue(org.card_complete(p))
            p.write_text(json.dumps({'photos':[{'file':'missing.jpg'}]}))
            self.assertFalse(org.card_complete(p))
            p.write_text(json.dumps({'photos':[{'status':'download_failed'}]}))
            self.assertFalse(org.card_complete(p))

    def test_valid_verdict(self):
        self.assertEqual(ws.parse_json(json.dumps(verdict())), verdict())

    def test_schema_rejects_invalid_fields(self):
        for key, value in [('overall', '0'), ('overall', True), ('overall', 4),
                           ('confidence', float('nan')), ('confidence', 2),
                           ('corners', []), ('photo_usable', 'true')]:
            v = verdict(); v[key] = value
            with self.subTest(key=key, value=value):
                self.assertIsNone(ws.parse_json(json.dumps(v)))

    def test_missing_usable_rejected(self):
        v = verdict(); del v['photo_usable']
        self.assertIsNone(ws.parse_json(json.dumps(v)))

    def test_unknown_location_cannot_be_slip(self):
        v = verdict(); v['corners']['TL'] = None
        self.assertFalse(ws.is_clean([verdict(), verdict(), v], 3, .6, 1))

    def test_hidden_heavy_damage_cannot_pass(self):
        v = verdict(); v['corners']['TL'] = 3
        self.assertFalse(ws.is_clean([verdict(), verdict(), v], 3, .6, 1))

    def test_real_light_slip_still_allowed(self):
        v = verdict(); v['overall'] = 1; v['corners']['TL'] = 1
        self.assertTrue(ws.is_clean([verdict(), verdict(), v], 3, .6, 1))
        self.assertFalse(ws.is_clean([verdict(), v, v], 3, .6, 1))

    def test_expired_state_retried_on_first_run(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'seen.json'
            p.write_text(json.dumps({'items': {'old': time.time()-40*86400}, 'images': {'oldimg': time.time()-40*86400}}))
            s = ef.State(p)
            self.assertFalse(s.seen_item('old'))
            self.assertFalse(s.seen_image('oldimg'))

    def test_known_item_does_not_hide_later_page(self):
        with tempfile.TemporaryDirectory() as d:
            s = ef.State(Path(d)/'seen.json'); s.mark('known')
            session = Mock()
            session.get.side_effect = [response({'itemSummaries':[item('known')], 'next': ef.SEARCH_URL+'?offset=200'}),
                                       response({'itemSummaries':[item('retry')]})]
            found, calls = ef.search_listings(session, 'fake', {'q':'pokemon','max_pages':2}, s, 'EBAY_US')
            self.assertEqual([x.item_id for x in found], ['retry'])

    def test_rejection_does_not_poison_other_search(self):
        with tempfile.TemporaryDirectory() as d:
            s = ef.State(Path(d)/'seen.json'); session=Mock()
            session.get.return_value=response({'itemSummaries':[item('candidate')]})
            ef.search_listings(session,'fake',{'q':'pokemon','exclude_keywords':['charizard']},s,'EBAY_US')
            found, _ = ef.search_listings(session,'fake',{'q':'charizard'},s,'EBAY_US')
            self.assertEqual(len(found),1)

    def test_variations_have_distinct_paths(self):
        self.assertNotEqual(ef.short_id('v1|123|456'),ef.short_id('v1|123|789'))
        self.assertEqual(ef.short_id('v1|123|0'),'123')

    def test_failed_download_manifest_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            m=org.process_card('abc',[('test',None)],Path(d),{})
            self.assertIs(m.get('complete'),False)

    def test_full_res_dedupes_image_variants(self):
        urls=org.listing_image_urls(Mock(),None,{'image_url':'https://i.ebayimg.com/images/g/test/s-l500.jpg',
                          'additional_images':['https://i.ebayimg.com/images/g/test/s-l1600.jpg']})
        self.assertEqual(len(urls),1)

    def test_model_feature_fallback_is_per_model(self):
        pr._openai_features.clear()
        pr._openai_features.update({'schema': True, 'reasoning': True})
        seen=[]
        def post(url, body, headers):
            seen.append(copy.deepcopy(body))
            if body['model']=='old' and 'reasoning_effort' in body:
                raise pr.APIError('400: unsupported reasoning_effort')
            return {'choices':[{'message':{'content':json.dumps(verdict())},'finish_reason':'stop'}]}
        with patch.dict(os.environ,{'OPENAI_API_KEY':'fake'}), patch.object(pr,'_post',post):
            pr.call_openai('old','test','test',[])
            pr.call_openai('new','test','test',[])
        self.assertIn('reasoning_effort',seen[-1])

    def test_api_error_does_not_expose_key(self):
        secret='sk-audit-test-secret'
        r=Mock(ok=False,status_code=401,text='Incorrect API key provided: '+secret)
        with patch.dict(os.environ,{'OPENAI_API_KEY':secret}),patch.object(pr.requests,'post',return_value=r):
            with self.assertRaises(pr.APIError) as e: pr._post('https://api.openai.com/v1/chat/completions',{}, {})
        self.assertNotIn(secret,str(e.exception))

if __name__=='__main__': unittest.main(verbosity=2)
