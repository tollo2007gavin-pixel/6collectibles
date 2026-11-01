"""Guarded GX source-track workflow. No yellow processing or colour thresholds."""
import copy
from .engine import analyze as seed_analyze,classify
from .tracks import compare_tracks,measure
from .anchored_polynomial import outer_tracks
from .horizontal_refinement import candidates,select_outer
from .keyline_selection import refine_observed_tracks
from .holder_topology import hypotheses,baseline_evidence
from .holder_repair import repair,audit_repair
from .quality_evidence import inspect_quality,reviewed_evidence
from .reportability import apply_reportability
from .exterior import apply_exterior,inspect_exterior
from .observed_recovery import recover_missing_inner,audit_inner_recovery
from .observed_extension import extend_incomplete_axes,audit_extension

def preferred_tracks(image,g,methods=None):
    methods=methods or compare_tracks(image,g)
    outer=outer_tracks(image,g);horizontal=candidates(image,g)
    for side,t in outer.items():
        if not t.get('supported'):outer[side]=copy.deepcopy(methods['source_curve']['tracks']['outer'][side])
    for side in ['top','bottom']:outer[side],_=select_outer(outer[side],horizontal[side]['outer_straight'])
    inner=copy.deepcopy(methods['source_curve']['tracks']['inner'])
    for side in ['top','bottom']:
        line=methods['source_straight']['tracks']['inner'][side];curve=inner[side]
        if line.get('supported') and (not curve.get('supported') or (curve.get('support',0)-line.get('support',0)<10/241 and line.get('coverage',0)>=.8*curve.get('coverage',0))):inner[side]=copy.deepcopy(line)
    return dict(outer=outer,inner=inner)

def refine(image,g,prior,*,seed_result=None,holder_evidence=None,check_stability=True):
    outer_recovery=[]
    if (seed_result or {}).get('status')=='unresolved':
        from .source_completion import extend_outer
        prior,outer_recovery=extend_outer(image,g,prior)
    original_prior=prior
    prior,rescued_inner=recover_missing_inner(image,g,prior)
    inner_audit=audit_inner_recovery(image,g,original_prior,prior,rescued_inner) if check_stability and rescued_inner else None
    keyline=refine_observed_tracks(image,g,prior,check_stability=check_stability)
    evidence=holder_evidence or dict(hypotheses=hypotheses(image,g),baseline=baseline_evidence(image,g))
    holder=repair(image,g,keyline['tracks'],evidence=evidence)
    if check_stability and holder.get('repairs'):
        holder['stability']=audit_repair(image,g,keyline['tracks'],holder)
        if holder.get('same_edge_agreement') and not holder['stability']['stable']:
            # A failed coverage enhancement must not withdraw an already valid
            # axis. No boundary identity moved; retain the previous tracks.
            declined=dict(repairs=holder['repairs'],stability=holder['stability'])
            holder=dict(geometry=copy.deepcopy(g),tracks=copy.deepcopy(keyline['tracks']),
                        axes=measure(keyline['tracks'],g['outer']),repairs=[],choices=[],review_required=False,
                        declined_same_edge_refinement=declined)
    from .final_frame import repair_certificate,certify_frame
    if holder.get('repairs'):
        holder['final_boundary_certificate']=repair_certificate(image,holder)
        # A repaired seam is no longer the selected edge. Reobserve the final
        # geometry; retain competing candidates and rounded-corner checks.
        refreshed=dict(hypotheses=hypotheses(image,holder['geometry']),
                       baseline=baseline_evidence(image,holder['geometry']))
        final_evidence=copy.deepcopy(evidence)
        for item in holder['repairs']:
            side=item['side']
            final_evidence['hypotheses'][side]=refreshed['hypotheses'][side]
            final_evidence['baseline'][side]=refreshed['baseline'][side]
    else:final_evidence=evidence
    auto=inspect_quality(image,holder['geometry'],holder['tracks'],final_evidence,holder)
    if auto.get('details',{}).get('holder_repair_source_sequence_verified'):
        holder['review_required']=False
    if check_stability and holder.get('repairs'):
        if not holder['stability']['stable']:auto['issues'].append('unstable_boundary_selection')
    before_extension=holder['tracks']
    extended,extensions=extend_incomplete_axes(image,holder['geometry'],before_extension)
    extension_audit=audit_extension(image,holder['geometry'],before_extension,extended,extensions) if check_stability and extensions else None
    if extensions:
        holder['tracks']=extended;holder['axes']=measure(extended,holder['geometry']['outer'])
    if extension_audit:
        for c in extension_audit['checks']:
            if not c['stable']:auto.setdefault('axis_issues',{}).setdefault(c['axis'],[]).append('unstable_boundary_selection')
    if keyline.get('selection_stable') is False:
        # This audit reruns only inner-bottom selection with a fixed frame.
        # It cannot invalidate the unchanged left/right observations.
        auto.setdefault('axis_issues',{})['tb']=['unstable_boundary_selection']
    if inner_audit:
        for check in inner_audit['checks']:
            if not check['stable']:
                axis='lr' if check['side'] in ['left','right'] else 'tb'
                auto.setdefault('axis_issues',{}).setdefault(axis,[]).append('unstable_boundary_selection')
    base=seed_result or {}
    # Drawing plausible tracks must never undo an unresolved localization,
    # cropped-exterior rejection, or an explicit review rejection.
    result=dict(geometry=holder['geometry'],tracks=holder['tracks'],axes=holder['axes'],
                diagnostic_axes=copy.deepcopy(holder['axes']),status=classify(holder['axes']),
                accepted=False,selection_gates_enabled=False,accuracy_calibrated=False,
                default_method='guarded_source_tracks',reasons=[],automatic_quality=auto,
                holder_repair=holder,keyline_selection=keyline,
                seed_status=base.get('status'),seed_reasons=base.get('reasons',[]))
    result['inner_recovery']=dict(sides=rescued_inner,stability=inner_audit)
    result['outer_recovery']=outer_recovery
    result['support_extension']=dict(changes=extensions,stability=extension_audit)
    frame_proof=None
    if base.get('status')=='unresolved':
        frame_proof=certify_frame(image,holder['geometry'],holder['tracks'],auto,final_evidence)
        result['final_frame_certificate']=frame_proof
    if base.get('status')=='hard_failure' or (base.get('status')=='unresolved' and not frame_proof['verified']):
        result['status']=base['status'];result['axes']={}
        result['reasons'].append('Original localization/physical-frame eligibility remains unresolved; candidate tracks are diagnostic.')
    elif frame_proof and frame_proof['verified']:
        result['reasons'].append('Early localization veto superseded by independently supported final physical boundaries.')
    result=apply_exterior(result,inspect_exterior(image,[holder['geometry']]))
    result=apply_reportability(result,auto)
    result['automatic_measurement_status']=result['status']
    result['automatic_measured_axes']={a:bool(result.get('axes',{}).get(a,{}).get('measured')) for a in ['lr','tb']}
    reviewed=reviewed_evidence(image,result)
    if reviewed:
        result['reviewed_quality']=reviewed
        result=apply_reportability(result,reviewed)
    result['margin_report_allowed']=result['status'] in ['complete','partial']
    return result

def analyze(image,*,holder='unknown',capture='photo',shadow=True,experimental_partials=False,quality_evidence=None):
    base=seed_analyze(image,holder=holder,capture=capture,shadow=True,experimental_partials=experimental_partials)
    g=base.get('geometry') or base.get('diagnostic_geometry')
    completed=None
    if (g is None and base['status']=='unresolved' and capture=='photo'
        and not base.get('localization',{}).get('ambiguous')):
        from .frame_completion import complete_frame
        completed,proof=complete_frame(image)
        base['localization_recovery']=proof
        if completed:
            g=completed['geometry']
            base.update(geometry=g,status='complete',reasons=['Frame recovered from agreeing source-observed GX boundaries'])
    if (g is None or base.get('localization',{}).get('ambiguous')) and base['status']=='unresolved' and capture=='photo':
        from .source_completion import recover_frame
        completed,proof=recover_frame(image,base.get('localization_candidates',[]))
        base['source_localization_recovery']=proof
        if completed:
            g=completed['geometry']
            base.update(geometry=g,status='complete',reasons=['Frame recovered from independently validated source boundaries'])
        elif base.get('localization',{}).get('ambiguous'):
            g=None
    if not g or base['status']=='hard_failure':
        reviewed=reviewed_evidence(image) if image is not None else None
        return apply_reportability(apply_reportability(base,reviewed),quality_evidence)
    result=refine(image,g,completed['tracks'] if completed else preferred_tracks(image,g,base.get('methods')),seed_result=base)
    if 'localization_recovery' in base:result['localization_recovery']=base['localization_recovery']
    if 'source_localization_recovery' in base:result['source_localization_recovery']=base['source_localization_recovery']
    return apply_reportability(result,quality_evidence)
