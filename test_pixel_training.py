import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pixel_training as t
from edge_classifier import DEFAULTS,VERSION,classify


def candidate(identifier=1,x=50,contrast=30):
    return dict(id=identifier,corner='TL',box_in_corner=[x,100,10,10],reason_code='',
        area_at_500px=80,grain=dict(status='measured',noise_sigma=1,contrast_p75=contrast,saturation_loss_p75=20),
        shape=dict(axis_ratio=1.2,fill_fraction=.8),
        edge=dict(status='agreement',possible_card_overlap=.8,uncertainty_fraction=1,min_outside_distance=0,glare_overlap=0))


def mark(identifier=1,x=50,label='whitening'):
    return dict(id=str(identifier),corner='TL',label=label,box=[x,100,10,10])


def card(number=1,split='development'):
    listing=f'v1|{1000+number}|0';sha=hashlib.sha256(str(number).encode()).hexdigest()
    result=dict(candidates=[candidate()],excluded_candidates=[],measurement_image_size=[1010,1050],
        source_sha256=sha,listing=listing,analysis_mode='whitening_classification',
        code_sha256={name:hashlib.sha256(Path(t.__file__).with_name(name).read_bytes()).hexdigest() for name in t.CODE_FILES})
    result['whitening_classification']=classify(result)
    return dict(number=number,listing=listing,source_sha256=sha,group_id=listing,split=split,
        annotations=[mark()],complete_corners=[],result=result)


class PixelTrainingTests(unittest.TestCase):
    def test_compressed_labels_load_identically(self):
        import gzip
        c=card()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'labels.json.gz'
            p.write_bytes(gzip.compress(json.dumps({'schema':t.SCHEMA,'cards':[c]}).encode()))
            self.assertEqual(t.load_dataset(p),t.validate_cards([c]))
            self.assertEqual(t.load_dataset(Path(directory)),t.validate_cards([c]))

    def test_positive_only_reviews_label_untouched_candidates_without_inventing_review(self):
        c=card();c['result']['candidates'].append(candidate(2,200));c['positive_only_review']=True
        checked=t.validate_cards([c])[0]
        self.assertEqual([a['label'] for a in checked['annotations']],['whitening','not_whitening'])
        self.assertEqual(t.score_cards([checked],DEFAULTS)['counts']['false_positive_marks'],1)
        c['annotations']=[mark(1,50,'uncertain')]
        checked=t.validate_cards([c])[0]
        self.assertEqual(checked['annotations'][0]['label'],'uncertain')
        c['annotations']=[]
        with self.assertRaises(ValueError):t.validate_cards([c])
        c['notes']='No whitening'
        self.assertEqual(len(t.validate_cards([c])[0]['annotations']),2)


    def test_partial_review_does_not_invent_negatives_and_complete_corner_does(self):
        c=card();c['result']['candidates'].append(candidate(2,200))
        partial=t.score_cards([c],DEFAULTS)
        self.assertEqual(partial['counts']['ignored_unreviewed_marks'],1)
        self.assertEqual(partial['counts']['false_positive_marks'],0)
        self.assertIsNone(partial['precision'])
        self.assertIsNone(partial['f1'])
        c['complete_corners']=['TL'];complete=t.score_cards([c],DEFAULTS)
        self.assertEqual(complete['counts']['false_positive_marks'],1)
        self.assertEqual(complete['precision'],.5)

    def test_uncertain_regions_never_become_false_alarms(self):
        c=card();c['result']['candidates'].append(candidate(2,200));c['complete_corners']=['TL']
        c['annotations'].append(mark(2,200,'uncertain'))
        r=t.score_cards([c],DEFAULTS)
        self.assertEqual(r['counts']['ignored_uncertain_marks'],1)
        self.assertEqual(r['counts']['false_positive_marks'],0)

    def test_fragments_of_one_chip_do_not_inflate_precision_or_false_alarms(self):
        c=card();c['result']['candidates'].append(candidate(2,52))
        r=t.score_cards([c],DEFAULTS)
        self.assertEqual(r['counts']['true_positive_regions'],1)
        self.assertEqual(r['counts']['duplicate_chip_marks'],1)
        self.assertEqual(r['counts']['false_positive_marks'],0)

    def test_one_prediction_cannot_claim_two_distinct_chips(self):
        c=card();c['annotations']=[dict(mark(),box=[50,100,4,10]),dict(mark(2),box=[56,100,4,10])]
        r=t.score_cards([c],DEFAULTS)
        self.assertEqual(r['counts']['true_positive_regions'],1)
        self.assertEqual(r['counts']['false_negative_regions'],1)

    def test_no_candidate_and_abstaining_candidate_have_different_miss_counts(self):
        c=card();c['result']['candidates'][0]['area_at_500px']=3;c['annotations'].append(mark(2,200))
        r=t.score_cards([c],DEFAULTS)
        self.assertEqual(r['counts']['false_negative_regions'],2)
        self.assertEqual(r['counts']['inconclusive_positive_regions'],1)
        self.assertEqual(r['counts']['extraction_missed_regions'],1)

    def test_patch_recovery_uses_supporting_box(self):
        c=card();a=c['result']['candidates'][0];a['box_in_corner']=[100,100,2,2];a['area_at_500px']=4
        a['shape']['patch_evidence']=dict(scales=[dict(scale=s,status='measured',area_at_500px=200,
            axis_ratio=1.2,fill_fraction=.8,edge_overlap=1,min_outside_distance=0,core_gap=2,
            contrast_p75=30,box_in_corner=[105,100,15,15]) for s in (8,12)])
        c['annotations']=[dict(mark(),box=[105,100,15,15])]
        self.assertEqual(t.score_cards([c],DEFAULTS)['counts']['true_positive_regions'],1)
        for patch in a['shape']['patch_evidence']['scales']:patch['contrast_p75']=10
        abstained=t.score_cards([c],DEFAULTS)
        self.assertEqual(abstained['counts']['extraction_missed_regions'],0)
        self.assertEqual(abstained['counts']['inconclusive_positive_regions'],1)

    def test_cross_split_aliases_and_transitive_aliases_rejected(self):
        for key in ('group_id','listing','source_sha256'):
            a=card();b=card(2,'validation');b[key]=a[key]
            if key in ('listing','source_sha256'):b['result'][key]=a[key]
            with self.subTest(key=key),self.assertRaises(ValueError):t.validate_cards([a,b])
        a=card();b=card(2);b['listing']=a['listing'];b['result']['listing']=a['listing']
        c=card(3,'validation');c['group_id']=b['group_id']
        with self.assertRaises(ValueError):t.validate_cards([a,b,c])

    def test_listing_urls_and_legacy_ids_are_one_identity(self):
        a=card();b=card(2,'validation');b['listing']='https://www.ebay.com/itm/some-card/1001?x=1';b['result']['listing']=b['listing']
        with self.assertRaises(ValueError):t.validate_cards([a,b])

    def test_coordinate_bounds_conflicting_labels_and_incompatible_cache_rejected(self):
        for problem in ('bounds','conflict','candidate_only','oldcode'):
            c=card()
            if problem=='bounds':c['annotations'][0]['box']=[499,499,5,5]
            if problem=='conflict':c['annotations'].append(mark(2,50,'not_whitening'))
            if problem=='candidate_only':c['result']['analysis_mode']='candidate_detection_only'
            if problem=='oldcode':c['result']['code_sha256']['edge_classifier.py']='0'*64
            with self.subTest(problem=problem),self.assertRaises(ValueError):t.validate_cards([c])

    def development(self):
        cards=[card(n) for n in (1,2,3)]
        for c in cards:
            c['result']['candidates'].append(candidate(2,200,21));c['annotations'].append(mark(2,200,'not_whitening'))
        return cards

    def test_tuning_ignores_changed_heldout_answers(self):
        cards=self.development()+[card(4,'validation')]
        first,config=t.calibrate(t.validate_cards(cards))
        cards[-1]['annotations']=[mark(55,400,'not_whitening')];cards[-1]['complete_corners']=['BR']
        second,other=t.calibrate(t.validate_cards(cards))
        self.assertEqual(first,second);self.assertEqual(config,other)
        self.assertEqual(first['selection_status'],'development_improvement')
        self.assertGreater(first['selected']['f1'],first['baseline']['f1'])
        self.assertNotIn('validation_cards',first)

    def test_too_little_data_freezes_baseline_without_claiming_tuning(self):
        report,config=t.calibrate([card()])
        self.assertEqual(report['selection_status'],'insufficient_labels')
        self.assertEqual(config['core_thresholds'],DEFAULTS)
        self.assertEqual(report['trials'],[])

    def test_frozen_validation_refuses_mutated_config_training_overlap_and_exposure_reuse(self):
        _,config=t.calibrate(self.development());reserved=card(4,'validation')
        report,exposure=t.validate_frozen([reserved],config)
        self.assertEqual(report['counts']['true_positive_regions'],1)
        with self.assertRaises(ValueError):t.validate_frozen([reserved],config,[exposure])
        bad=copy.deepcopy(config);bad['core_thresholds']['min_area']=1
        with self.assertRaises(ValueError):t.validate_frozen([reserved],bad)
        bad=card(1,'validation')
        with self.assertRaises(ValueError):t.validate_frozen([bad],config)

    def test_directory_merge_skips_manifests_deduplicates_and_loads_ledger(self):
        c=card();_,config=t.calibrate([c]);_,exposure=t.validate_frozen([card(2,'validation')],config)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,payload in [('one.json',{'schema':t.SCHEMA,'cards':[c]}),('copy.json',{'schema':t.SCHEMA,'cards':[c]}),
                                 ('config.json',config),('exposure.json',exposure)]:
                (root/name).write_text(json.dumps(payload))
            self.assertEqual(len(t.load_dataset(root)),1)
            self.assertEqual(t.load_exposures(root),[exposure])
            self.assertEqual(t.load_exposures(root/'one.json'),[exposure])
            c['annotations'][0]['label']='not_whitening';(root/'copy.json').write_text(json.dumps({'schema':t.SCHEMA,'cards':[c]}))
            with self.assertRaises(ValueError):t.load_dataset(root)


if __name__=='__main__':unittest.main()
