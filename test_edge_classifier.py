import copy,unittest
from edge_classifier import classify

def candidate(identifier=1,corner='TR',box=None):
    return dict(id=identifier,corner=corner,box_in_corner=box or [100,100,10,10],reason_code='',
                area_at_500px=80,grain=dict(status='measured',noise_sigma=3,contrast_p75=28,saturation_loss_p75=20),
                shape=dict(axis_ratio=1.5,fill_fraction=.6),
                edge=dict(status='agreement',possible_card_overlap=.8,uncertainty_fraction=1,
                          min_outside_distance=0,glare_overlap=0))

def run(items,size=500):
    return classify(dict(candidates=items,excluded_candidates=[],measurement_image_size=[2*size+10,2*size+50]))['decisions']

class EdgeClassifierTests(unittest.TestCase):
    def test_strong_patch_and_detached_sleeve_differ_without_labels(self):
        a=candidate();b=candidate(2)
        b['edge'].update(possible_card_overlap=0,min_outside_distance=8)
        self.assertEqual([d['classification'] for d in run([a,b])],['likely_whitening','likely_artifact'])

    def test_uncertain_is_not_clean(self):
        for field,value in [('shape',dict(axis_ratio=9,fill_fraction=.7)),
                            ('grain',dict(status='unstable_reference')),
                            ('edge',dict(status='unconfirmed'))]:
            c=candidate();c[field]=value
            self.assertEqual(run([c])[0]['classification'],'inconclusive')
        c=candidate();c['grain']['noise_sigma']=12
        self.assertEqual(run([c])[0]['classification'],'inconclusive')

    def test_triangle_rejected_neighbor_can_be_recovered_but_not_an_entire_chain(self):
        seed=candidate();near=candidate(2,box=[118,100,10,10]);far=candidate(3,box=[136,100,10,10])
        for c in (near,far):c['edge']['uncertainty_fraction']=0;c['reason_code']='T'
        result=run([far,near,seed])
        self.assertEqual([d['classification'] for d in result],['likely_whitening','likely_whitening','inconclusive'])
        self.assertEqual(result[1]['supporting_candidates'],[1])
        near['corner']='TL'
        self.assertEqual(run([seed,near])[1]['classification'],'inconclusive')

    def test_large_weak_spot_has_no_exception(self):
        c=candidate();c['area_at_500px']=1000;c['grain']['contrast_p75']=15
        self.assertEqual(run([c])[0]['classification'],'inconclusive')

    def test_ids_order_and_human_labels_are_not_features(self):
        a=candidate();b=candidate(2);b['shape']['axis_ratio']=9
        first=run([a,b]);a['id']=900;b['id']=800
        a['human_label']='not clean';b['human_label']='clean'
        second=run([b,a]);self.assertEqual(first[0]['classification'],second[1]['classification'])
        self.assertEqual(first[1]['classification'],second[0]['classification'])

    def test_neighbor_distance_normalizes_with_scale(self):
        a=candidate();b=candidate(2,box=[118,100,10,10]);b['edge']['uncertainty_fraction']=0
        before=run([a,b]);scaled=copy.deepcopy([a,b])
        for c in scaled:c['box_in_corner']=[v*2 for v in c['box_in_corner']]
        after=run(scaled,1000)
        self.assertEqual([d['classification'] for d in before],[d['classification'] for d in after])

if __name__=='__main__':unittest.main()
