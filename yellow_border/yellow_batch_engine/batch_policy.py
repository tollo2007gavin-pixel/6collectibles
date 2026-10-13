"""Validated batch measurement policy; no centered-card selection cutoff."""
import math
from .compare import compare,DEFAULT_GATES


def analyze_front(image,*,measurement_frame,perspective_limit):
    # The eBay runner is serial. Scope the legacy shared gate to this call so
    # both recovery modules see the same limit, including on failure paths.
    if not math.isfinite(perspective_limit) or not 0<perspective_limit<=1:
        raise ValueError('perspective_limit must be in (0, 1]')
    previous=DEFAULT_GATES['keystone']
    try:
        DEFAULT_GATES['keystone']=perspective_limit
        return compare(image,measurement_frame=measurement_frame)
    finally:
        DEFAULT_GATES['keystone']=previous


def describe_measurement(analysis,combined,*,review_threshold=.15):
    scores={};allowances={};ranges={}
    for axis,value in combined['axes'].items():
        selected=[]
        if value['resolved']:
            for method in analysis['methods'].values():
                a=method['axes'][axis]
                if a['resolved']:
                    index=a.get('selected_trial')
                    if index is not None:
                        selected.append(method['trials'][index]['seed']['keystone'])
        score=max(selected) if selected else None
        allowance=(1. if score>review_threshold else 0.) if score is not None else None
        scores[axis]=score;allowances[axis]=allowance
        ranges[axis]=[max(0.,value['percentage']-allowance),min(100.,value['percentage']+allowance)] if value['resolved'] and allowance is not None else None
    if combined['status']=='Fail':category='fail'
    elif not all(a['resolved'] for a in combined['axes'].values()):category='undetermined'
    else:category='pass/review' if any(v is None or v>0 for v in allowances.values()) else 'pass'
    return dict(category=category,axis_perspective=scores,axis_allowance_pp=allowances,
        axis_percentage_range=ranges,range_kind='proposed_angular_allowance',statistical_confidence_level=None,
        centering_selection_gate_applied=False)
