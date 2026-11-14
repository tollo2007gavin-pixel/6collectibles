"""Experimental, selective whitening classification. No labels or IDs as features."""
import math

VERSION='edge-evidence-v1'
DEFAULTS=dict(min_area=40,max_axis_ratio=3,min_fill=.25,contrast_floor=20,noise_multiple=4,
              saturation_loss=15,strong_contrast=32,strong_noise_multiple=5,
              min_card_overlap=.2,min_edge_overlap=.25,neighbor_fraction=.024)


def signal(candidate,limits=None):
    t=DEFAULTS if limits is None else limits
    g=candidate.get('grain',{});s=candidate.get('shape',{})
    if g.get('status')!='measured':return False,'unreliable_local_color_reference'
    noise=g.get('noise_sigma',float('inf'))
    contrast=g.get('contrast_p75',0);loss=g.get('saturation_loss_p75',0)
    if candidate['area_at_500px']<t['min_area']:return False,'too_small_for_strong_evidence'
    if s.get('axis_ratio',float('inf'))>t['max_axis_ratio']:return False,'elongated_mark_could_be_glare_or_damage'
    if s.get('fill_fraction',0)<t['min_fill']:return False,'fragmented_patch'
    if contrast<max(t['contrast_floor'],t['noise_multiple']*noise):return False,'brightness_change_not_distinct_from_local_noise'
    if loss<t['saturation_loss'] and contrast<max(t['strong_contrast'],t['strong_noise_multiple']*noise):return False,'weak_color_change'
    return True,'coherent_light_patch_above_local_noise'


def gap(a,b):
    x,y,w,h=a['box_in_corner'];u,v,p,q=b['box_in_corner']
    return math.hypot(max(0,x-(u+p-1),u-(x+w-1)),max(0,y-(v+q-1),v-(y+h-1)))


def classify(result,limits=None):
    t=dict(DEFAULTS if limits is None else limits)
    candidates=sorted(result['candidates']+result['excluded_candidates'],key=lambda c:c['id'])
    tile=(result['measurement_image_size'][0]-10)/2
    decisions=[];seeds=[];eligible=[]
    for c in candidates:
        e=c.get('edge',{});code=c.get('reason_code','')
        d=dict(id=c['id'],corner=c['corner'],classification='inconclusive',reason='',
               detector_filter=code or 'retained',box_in_corner=c['box_in_corner'])
        if code in ('A','B'):
            d.update(classification='likely_artifact',reason='printed_artwork' if code=='A' else 'outside_protected_card_region')
        elif e.get('status')!='agreement':d['reason']='unreliable_edge_location'
        elif e.get('possible_card_overlap',0)==0 and e.get('min_outside_distance',0)>=tile*.01 and e.get('glare_overlap',0)<.5:
            d.update(classification='likely_artifact',reason='detached_from_all_card_estimates')
        else:
            strong,why=signal(c,t);d['reason']=why
            if strong and e.get('possible_card_overlap',0)>=t['min_card_overlap']:
                if e.get('uncertainty_fraction',0)>=t['min_edge_overlap']:
                    d.update(classification='likely_whitening',reason='strong_card_associated_edge_patch')
                    seeds.append(c)
                else:
                    d['reason']='strong_interior_patch_requires_edge_connection'
                    eligible.append((c,d))
        decisions.append(d)
    # One-hop only: a strong interior patch may be part of a nearby edge chip.
    # No size exemption and no recursive chaining through weaker interior marks.
    for c,d in eligible:
        neighbors=[s for s in seeds if s['corner']==c['corner'] and gap(c,s)<=tile*t['neighbor_fraction']]
        if neighbors:
            d.update(classification='likely_whitening',reason='strong_patch_adjacent_to_supported_edge_damage',
                     supporting_candidates=[s['id'] for s in neighbors])
    return dict(version=VERSION,thresholds=t,decisions=decisions,
                counts={name:sum(d['classification']==name for d in decisions)
                        for name in ('likely_whitening','likely_artifact','inconclusive')},
                warning='Experimental evidence rules; no calibrated accuracy or confidence. Inconclusive is not clean.')
