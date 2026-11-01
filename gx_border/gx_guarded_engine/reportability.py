"""Withhold reportable margins when boundary evidence is known unreliable.

Evidence may come from an explicit visual review or a validated detector. This
module consumes evidence; it does not claim to detect glare, holders or pose.
These are measurement-quality gates, not centering-ratio selection gates.
"""
import copy

BLOCKERS=frozenset({
    'unreliable_lighting','unreliable_perspective','insufficient_boundary_detail',
    'holder_boundary_ambiguity','unresolved_boundary_identity','unsupported_layout',
    'unstable_boundary_selection',
})


def apply_reportability(result, evidence=None):
    """Suppress affected axes; whole-photo issues still suppress both.

    Missing evidence is not approval and this never promotes a previous reject.
    A holder's mere presence is not a blocker: confusion about its edges is.
    """
    if evidence is None:return result
    supplied=set(evidence.get('issues',[]))
    axis_issues=evidence.get('axis_issues',{})
    if set(axis_issues)-{'lr','tb'}:raise ValueError('Unknown measurement axis')
    unknown=(supplied|set().union(*[set(v) for v in axis_issues.values()]))-BLOCKERS
    if unknown:raise ValueError('Unknown measurement-quality issue: '+', '.join(sorted(unknown)))
    updated=copy.deepcopy(result)
    by_axis={a:sorted(supplied|set(axis_issues.get(a,[]))) for a in ['lr','tb']}
    blocking=sorted(supplied|set().union(*[set(v) for v in axis_issues.values()]))
    if not blocking:return updated
    previous=updated.get('reportability',{})
    blocked_axes=set(previous.get('blocked_axes',[]))|{a for a,v in by_axis.items() if v}
    if previous.get('blocked'):blocked_axes|={'lr','tb'}
    updated['reportability']=dict(blocked=len(blocked_axes)==2,issues=sorted(set(blocking)|set(previous.get('issues',[]))),
        blocked_axes=sorted(blocked_axes),axis_issues=by_axis,
        source=evidence.get('source','unspecified'),automatic_detection=bool(evidence.get('automatic_detection',False)),
        centering_selection_gate=False)
    updated.setdefault('diagnostic_axes',copy.deepcopy(updated.get('axes',{})))
    for a,issues in by_axis.items():
        if issues:updated.setdefault('axes',{})[a]=dict(measured=False,reason='measurement quality withheld',issues=issues)
    count=sum(bool(a.get('measured')) for a in updated.get('axes',{}).values())
    # Suppression can never promote a previously unresolved or failed frame.
    if result.get('status') in ['complete','partial']:
        updated['status']={0:'unresolved',1:'partial',2:'complete'}[count]
    updated['accepted']=False
    updated['margin_report_allowed']=updated.get('status') in ['complete','partial'] and count>0
    updated['reasons']=list(updated.get('reasons',[]))+['Margins withheld: '+', '.join(blocking)]
    return updated
