"""Isolated GX review adapter. No centering selection or grading acceptance."""
import cv2
import numpy as np
from .production_gx import GXImageAssumptions, estimate_full_art_gx
from .legacy.frame_holders import measure_holder, validate_card_geometry
from .legacy.frame_recovery import recover_frame
from .legacy.frame_keyline_segments import refine_vertical_keylines
from .tracks import compare_tracks
from .bare import measure_bare
from .outer_extent import apply_outer_extent_guard
from .exterior import inspect_exterior, apply_exterior
from .reportability import apply_reportability


def axis_ratios(centering):
    return {axis:dict(measured=True,first_pct=centering[key]) for axis,key in [('lr','left_pct'),('tb','top_pct')]}


def classify(axes, *, hard_failure=False, ambiguous=False):
    if hard_failure:return 'hard_failure'
    if ambiguous:return 'unresolved'
    count=sum(bool(axes.get(a,{}).get('measured')) for a in ['lr','tb'])
    return {2:'complete',1:'partial',0:'unresolved'}[count]


def retain_inner_only(prior, candidate):
    """Permit structural inner repair only when fusion retained every outer side.

    Whole-frame fusion has not reproduced saved correct geometry reliably.
    Never let this limited port replace a previously supported physical edge.
    """
    if candidate.get('replace_outer') or candidate.get('replaced_sides'):return prior
    outer=np.asarray(prior['outer']);width=np.linalg.norm(outer[1]-outer[0])
    if np.max(np.linalg.norm(np.asarray(candidate['outer'])-outer,axis=1))>width*1e-5:return prior
    movement=np.max(np.linalg.norm(np.asarray(candidate['inner'])-prior['inner'],axis=1))*1260/width
    if movement>14:return prior
    # Retain the original outer coordinates bit for bit.
    from .production_gx import centering_from_frame_quads
    return dict(candidate,outer=prior['outer'],centering=centering_from_frame_quads(outer,np.asarray(candidate['inner'])),
                accepted_outer_preserved=True)


def partial_candidate(image,candidates):
    """Only salvage an axis with four supported lines and supported outer frame.

    No bypass of ambiguous holder selection. Close alternatives must agree on
    all physical edges and the retained axis's two keylines before tracks run.
    """
    valid=[]
    for c in candidates:
        try:validate_card_geometry(image,c['outer'],c['inner'],c['stripe'])
        except (ValueError,cv2.error):continue
        support=c['support']
        if min(support[s+'_outer'] for s in ['left','right','top','bottom'])<.65 or min(c['stripe_support'].values())<.55:continue
        axes=[a for a,ss in [('lr',['left','right']),('tb',['top','bottom'])] if all(support[s+'_inner']>=.65 for s in ss)]
        if not axes:continue
        score=sum(support.values())/len(support);valid.append((score,c,axes))
    if not valid:return None,[]
    valid.sort(key=lambda x:-x[0]);score,best,axes=valid[0]
    width=np.linalg.norm(np.array(best['outer'])[1]-best['outer'][0]);retained=[]
    for axis in axes:
        side_indices=[(0,3),(1,2)] if axis=='lr' else [(0,1),(3,2)]
        agrees=True
        for s,c,other_axes in valid:
            if s<score-.12:continue
            if axis not in other_axes:agrees=False;break
            differences=[np.max(np.linalg.norm(np.array(c['outer'])-best['outer'],axis=1))]
            for ix in side_indices:differences.append(np.max(np.linalg.norm(np.array(c['inner'])[list(ix)]-np.array(best['inner'])[list(ix)],axis=1)))
            if max(differences)*1260/width>3:agrees=False;break
        if agrees:retained.append(axis)
    return (best if retained else None),retained


def analyze(image, *, holder='unknown', capture='photo', shadow=True, experimental_partials=False, quality_evidence=None):
    result=dict(accepted=False,selection_gates_enabled=False,accuracy_calibrated=False,
                default_method='legacy_segment',holder=holder,capture=capture,axes={},reasons=[],methods={})
    if image is None or image.ndim!=3 or min(image.shape[:2])<300:
        return dict(result,status='hard_failure',reasons=['Unreadable image or resolution below 300 pixels'])
    geometry=None;partial_axes=[];candidates=[]
    try:
        if capture=='scan' or holder=='none':
            try:
                measurement,quality=estimate_full_art_gx(image,GXImageAssumptions(capture=capture,holder='none'))
                result['production_quality']=quality
                if quality['accepted']:
                    geometry=dict(outer=measurement.outer_quad.tolist(),inner=measurement.inner_quad.tolist(),centering=measurement.centering)
                    if capture=='photo':
                        try:geometry=retain_inner_only(geometry,measure_bare(image,measurement,quality))
                        except (ValueError,cv2.error,np.linalg.LinAlgError) as error:result['reasons'].append('Inner-only fusion unavailable: '+str(error))
            except (ValueError,cv2.error) as error:result['reasons'].append(str(error))
            if geometry is None and capture=='photo':
                geometry=recover_frame(image,None,known_bare=True)
            if geometry is not None and capture=='photo' and 'segment_diagnostics' not in geometry:geometry=refine_vertical_keylines(image,geometry,known_bare=True)
        else:
            measured=measure_holder(image,holder=holder,capture=capture,include_candidates=True)
            candidates=measured.pop('diagnostic_candidates',[])
            result['localization_candidates']=candidates
            # Retain inspectable candidates without promoting weak evidence to
            # measured geometry or centering. The batch renderer labels these.
            diagnostic=[]
            for c in candidates:
                try:validate_card_geometry(image,c['outer'],c['inner'],c['stripe'])
                except (ValueError,cv2.error):continue
                diagnostic.append(c)
            if diagnostic:
                diagnostic.sort(key=lambda c:-float(np.mean(list(c['support'].values()))))
                result['diagnostic_geometry']=diagnostic[0]
                result['diagnostic_weak_sides']=[k for k,v in diagnostic[0]['support'].items() if v<.65]
            result['localization']=measured
            if measured.get('geometry_supported'):geometry=measured
            elif not measured.get('ambiguous'):
                geometry,partial_axes=partial_candidate(image,candidates)
                if geometry and experimental_partials:result['default_method']='source_straight_partial'
            if geometry is None:result['reasons']+=measured['reasons']
        if geometry is not None:
            result['geometry']=geometry
            if shadow or partial_axes:result['methods']=compare_tracks(image,geometry)
            if partial_axes:
                values=result['methods']['source_straight']['initial_frame']
                result['partial_proposal']={a:values[a] if a in partial_axes else dict(measured=False,reason='unsupported keyline on this axis') for a in ['lr','tb']}
                if experimental_partials:result['axes']=result['partial_proposal']
                result['reasons'].append('Partial diagnostic measurement; original whole-frame gate rejected')
            else:result['axes']=axis_ratios(geometry['centering'])
    except (ValueError,cv2.error,np.linalg.LinAlgError) as error:
        result['reasons'].append(str(error))
    if holder=='unknown':result['reasons'].append('GX layout, front orientation and holder eligibility require visual review')
    result['status']=classify(result['axes'])
    result=apply_outer_extent_guard(image,result)
    result=apply_exterior(result,inspect_exterior(image,[geometry] if geometry else candidates))
    return apply_reportability(result,quality_evidence)
