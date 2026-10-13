"""Offline tests: no credentials, HTTP requests or project-output deletion."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from PIL import Image
import ebay_line_fit_test as runner


def detail(condition='4000', card_condition='Near mint or better', **kw):
    return dict(conditionId=condition, conditionDescriptors=[dict(name='Card Condition',
        values=[dict(content=card_condition)])], image={'imageUrl': 'https://example.org/front.jpg'}, **kw)


class FakeClient:
    def __init__(self): self.downloaded=[]; self.searches=[]; self.detail_ids=[]
    def search(self, query, **kw):
        self.searches.append((query,kw))
        if query == 'first':
            ids=['previous','graded','played','a'] if kw['offset']==0 else ['b','extra']
        else: ids=['a','c','d']
        return dict(itemSummaries=[dict(itemId=i, additionalImages=[{'imageUrl':'https://example.org/back.jpg'}]) for i in ids],
                    next='next' if query=='first' and kw['offset']==0 else None)
    def details(self,item_id):
        self.detail_ids.append(item_id)
        if item_id=='graded': return detail(condition='2750')
        if item_id=='played': return detail(card_condition='Excellent')
        return detail(title=item_id, additionalImages=[{'imageUrl':'https://example.org/back.jpg'}])
    def download(self,url):
        self.downloaded.append(url)
        data=io.BytesIO(); Image.new('RGB',(200,280),'yellow').save(data,format='PNG')
        return data.getvalue()


def no_fit(image):
    from yellow_batch_engine.compare import unresolved
    return {'methods':{m:dict(status='No fit',axes=unresolved('Synthetic no-fit fixture'),
         paths={k:[None]*4 for k in ('outer','inner')},trials=[]) for m in runner.METHOD_FOLDERS}}


class BatchTests(unittest.TestCase):
    def test_background_gate_requires_three_sides_not_three_missing_fits(self):
        import numpy as np
        from yellow_batch_engine.exterior_gate import inspect_exterior_background
        full = np.full((240,180,3),255,np.uint8)
        full[35:205,30:150] = (40,210,245)  # BGR yellow card border.
        full[42:198,37:143] = (180,50,20)   # Artwork.
        analysis = no_fit(None)
        for image,expected in [(full,0),(full[:,30:150],2),(full[35:,30:150],3),(full[35:205,30:150],4)]:
            gate=inspect_exterior_background(image,analysis)
            self.assertEqual(gate['unavailable_count'],expected)
            self.assertEqual(gate['fail'],expected>=3)
        rotated=np.rot90(full[35:,30:150]).copy()
        self.assertTrue(inspect_exterior_background(rotated,analysis)['fail'])

    def test_yellow_surroundings_are_not_assumed_to_be_cropped_borders(self):
        import numpy as np
        from yellow_batch_engine.exterior_gate import inspect_exterior_background
        image=np.full((240,180,3),(30,200,250),np.uint8)
        image[55:185,50:130]=(60,215,230)
        image[61:179,56:124]=(180,50,20)
        self.assertFalse(inspect_exterior_background(image,no_fit(None))['fail'])

    def test_hard_fail_blocks_combined_ratios_but_preserves_paths(self):
        from yellow_batch_engine.exterior_gate import apply_exterior_gate
        from yellow_batch_engine.compare import combine_methods
        methods=no_fit(None)['methods']
        gate=dict(fail=True,unavailable_count=3,unavailable_sides=['top','right','left'],
                  reason='Fail: three exterior sides unavailable',code='insufficient_exterior_background')
        for mode in methods:
            methods[mode]['axes']={a:dict(resolved=True,reasons=[],percentage=50) for a in ('lr','tb')}
            methods[mode]['paths']['outer'][0]=[[1,2],[3,4]]
            methods[mode]=apply_exterior_gate(methods[mode],gate)
        combined=combine_methods(methods)
        self.assertEqual(combined['status'],'Fail')
        self.assertTrue(combined['image_gate']['fail'])
        self.assertTrue(all(not a['resolved'] and a['percentage'] is None for a in combined['axes'].values()))
        self.assertEqual(combined['paths']['outer'][0],[[1,2],[3,4]])

    def test_rarity_filter_only_for_ditto_csv_id(self):
        client = runner.EbayClient.__new__(runner.EbayClient)
        client.api = Mock()
        client.api.get.return_value.json.return_value = {}
        for card_id in ('39', '28', '22', None):
            client.search('query', offset=200, limit=20, age_hours=24, card_id=card_id)
            params = client.api.get.call_args.kwargs['params']
            self.assertEqual(params['offset'], 200)
            self.assertIn('conditionIds:{4000}', params['filter'])
            self.assertIn('itemStartDate:', params['filter'])
            if card_id in ('39', '28'):
                self.assertEqual(params['aspect_filter'], 'categoryId:183454,Rarity:{Ultra Rare}')
            else:
                self.assertNotIn('aspect_filter', params)

    def test_strict_conditions(self):
        self.assertTrue(runner.is_near_mint_ungraded(detail()))
        self.assertTrue(runner.is_near_mint_ungraded(detail(card_condition='Near Mint or Better')))
        for record in [detail(condition='2750'),detail(card_condition='Excellent'),
                       {'conditionId':'4000','title':'NM mint PSA 10'},detail(condition='')]:
            self.assertFalse(runner.is_near_mint_ungraded(record))

    def test_csv_only_active(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cards.csv'; path.write_text('id,search_query,active\n1,active,1\n2,disabled,0\n3,,1\n')
            self.assertEqual([r['id'] for r in runner.load_cards(path)],['1'])

    def test_replaces_only_owned_folders_and_preserves_previous_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'batch'; unrelated=Path(d)/'other'; unrelated.mkdir()
            (unrelated/'keep.txt').write_text('keep')
            with runner.BatchOutput(root):
                (root/'Combined'/'old.jpg').write_text('old')
                (root/'Forced_curve').mkdir()
                (root/'Forced_curve'/'legacy.jpg').write_text('old')
                runner.write_json(root/'batch.json',{'records':[{'item_id':'old'}]})
            with runner.BatchOutput(root) as output:
                self.assertEqual(output.previous_ids,{'old'})
                self.assertEqual(list((root/'Combined').iterdir()),[])
                self.assertFalse((root/'Forced_curve').exists())
            self.assertEqual((unrelated/'keep.txt').read_text(),'keep')

    def test_refuses_unowned_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'keep.txt').write_text('keep')
            with self.assertRaises(ValueError):
                with runner.BatchOutput(root): pass
            self.assertTrue((root/'keep.txt').exists())

    def test_concurrent_run_cannot_clear_first(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'batch'
            with runner.BatchOutput(root):
                (root/'Combined'/'keep.jpg').write_text('keep')
                with self.assertRaises(RuntimeError):
                    with runner.BatchOutput(root): pass
                self.assertTrue((root/'Combined'/'keep.jpg').exists())

    def test_link_guard_checks_before_delete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'batch'
            with runner.BatchOutput(root):
                (root/'Combined'/'keep.jpg').write_text('keep')
            real=runner.linked
            with patch.object(runner,'linked',side_effect=lambda p:p.name=='adaptive' or real(p)):
                with self.assertRaises(ValueError):
                    with runner.BatchOutput(root): pass
            self.assertTrue((root/'Combined'/'keep.jpg').exists())
            self.assertFalse((root/'.running').exists())

    def test_pagination_caps_dedup_and_main_image_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'batch'; client=FakeClient()
            with runner.BatchOutput(root) as output:
                output.previous_ids={'previous'}
                batch=runner.run_batch([{'id':'1','search_query':'first'},{'id':'2','search_query':'second'}],
                    client,output,total=3,per_card=2,candidates_per_card=400,age_hours=None,
                    processor=lambda content,meta,root:runner.process_front(content,meta,root,analyzer=no_fit))
                self.assertEqual([r['item_id'] for r in batch['records']],['a','b','c'])
                self.assertEqual([s[1]['card_id'] for s in client.searches], ['1', '1', '2'])
                self.assertNotIn('previous',client.detail_ids)
                self.assertEqual(len(client.downloaded),3)
                self.assertTrue(all(u.endswith('/front.jpg') for u in client.downloaded))
                self.assertEqual(len(list((root/'Combined').glob('*.jpg'))),3)
                self.assertFalse((root/'Forced_curve').exists())
                self.assertFalse((root/'adaptive').exists())
                self.assertEqual(len(list((root/'downloads').glob('*.png'))),3)
                with (root/'measurements.csv').open(encoding='utf-8-sig') as handle:
                    rows=list(csv.DictReader(handle))
                self.assertEqual(len(rows),3)
                self.assertTrue(all(r['left_pct']=='' for r in rows))

    def test_analysis_exception_still_produces_one_labelled_image(self):
        with tempfile.TemporaryDirectory() as d:
            with runner.BatchOutput(Path(d)/'batch') as output:
                def broken(image): raise ValueError('bad geometry')
                row=runner.process_front(FakeClient().download('front'),dict(card_id='1',query='q',item_id='x',title='x',url=''),output.root,analyzer=broken)
                self.assertTrue(all(r['status']=='Analysis error' for r in row['methods'].values()))
                self.assertTrue(all((output.root/p).exists() for p in row['overlays'].values()))
                self.assertEqual(list(row['overlays']), ['combined'])

    def test_combined_averages_valid_percentages_and_prefers_forced_paths(self):
        from yellow_batch_engine.compare import combine_methods
        methods = no_fit(None)['methods']
        for mode,values in [('adaptive', (40,60)), ('forced', (42,64))]:
            methods[mode]['axes'] = {axis: dict(resolved=True, reasons=[], percentage=value)
                                    for axis,value in zip(('lr','tb'),values)}
        methods['adaptive']['paths']['outer'][0] = [[1,2],[3,4]]
        methods['forced']['paths']['outer'][0] = [[5,6],[7,8]]
        methods['adaptive']['paths']['inner'][1] = [[9,10],[11,12]]
        result = combine_methods(methods)
        self.assertEqual(result['axes']['lr']['percentage'], 41)
        self.assertEqual(result['axes']['tb']['percentage'], 62)
        self.assertEqual(result['axes']['lr']['measurement_source'], 'mean')
        self.assertEqual(result['paths']['outer'][0], [[5,6],[7,8]])
        self.assertEqual(result['path_sources']['inner'][1], 'adaptive')
        self.assertIsNone(result['paths']['inner'][0])

    def test_combined_never_averages_rejected_measurements(self):
        from yellow_batch_engine.compare import combine_methods
        methods = no_fit(None)['methods']
        methods['forced']['axes']['lr'] = dict(resolved=True, reasons=[], percentage=47)
        methods['adaptive']['axes']['lr']['percentage'] = 80
        methods['adaptive']['axes']['tb'] = dict(resolved=True, reasons=[], percentage=53)
        result = combine_methods(methods)
        self.assertEqual(result['axes']['lr']['percentage'],47)
        self.assertEqual(result['axes']['lr']['measurement_source'],'forced')
        self.assertEqual(result['axes']['tb']['percentage'],53)
        self.assertEqual(result['axes']['tb']['measurement_source'],'adaptive')
        methods['adaptive']['axes']['tb']['resolved'] = False
        self.assertFalse(combine_methods(methods)['axes']['tb']['resolved'])


if __name__=='__main__': unittest.main()
