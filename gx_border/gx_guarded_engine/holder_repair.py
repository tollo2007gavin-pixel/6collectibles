"""Source-observed repair proposals for holder seams; no card/image lookup."""
import copy,cv2,numpy as np
from .holder_topology import FRAME,hypotheses,baseline_evidence
from .tracks import SIDES,project,extract,fit,measure,rebuild_frame


def same_edge_group(geometry,tracks,choices):
    """Jointly extend supported curves, without relocating boundary identity.

    Keep previously complete measurements unchanged. A group must agree with
    every existing observed fragment, retain all inner lines and support a new
    frame from all four observed physical edges.
    """
    from .quality_evidence import disagreement
    if not choices or any(c['kind']!='same_edge_curve' for c in choices):return None
    if all(a.get('measured') for a in measure(tracks,geometry['outer']).values()):return None
    q=np.asarray(geometry['outer']);scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    updated=copy.deepcopy(tracks);proof={}
    for choice in choices:
        side=choice['side'];k=SIDES.index(side);a,b=q[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
        candidate=choice['candidate']['track'];c=disagreement(tracks['outer'][side],candidate,a,b,n,scale)
        if c['common_profiles']<54 or c.get('p95_normalized_px',float('inf'))>3:return None
        proof[side]=c;updated['outer'][side]=copy.deepcopy(candidate)
    frame=rebuild_frame(updated,q)
    if frame is None:return None
    g=copy.deepcopy(geometry);g['outer']=frame.tolist()
    return dict(geometry=g,tracks=updated,axes=measure(updated,g['outer']),same_edge_agreement=proof,
                repairs=[dict(side=c['side'],kind='same_edge_curve',offset_normalized_px=c['candidate']['offset'],inner_action='retained') for c in choices],
                choices=choices,review_required=True)

def inner_from_physical_edge(image,outer,k):
    """Search GX signature modes inward from a repaired edge, not its old seam.

    The dark band and bright shoulder still determine the boundary. The 40 px
    offset only centers a broad search; it is never used as a measured margin.
    """
    q=np.asarray(outer,float);a,b=q[[k,(k+1)%4]];v=b-a
    n=np.array([-v[1],v[0]])/np.linalg.norm(v)
    scale=(np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2520
    seed=q.copy();seed[[k,(k+1)%4]]+=40*scale*n
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
    c,aa,bb,normal=extract(lab,seed,k,'inner_colour',scale,allow_partial_profiles=True,search_radius=32)
    if len(c)<20:return []
    hist=np.zeros(65)
    for row,t,d,w in c:
        if .2<t<.8:hist[int(np.clip(round(d)+32,0,64))]+=w
    modes=[];results=[]
    for index in np.argsort(hist)[::-1]:
        if hist[index]<3:break
        mode=float(index-32)
        if any(abs(mode-m)<6 for m in modes):continue
        modes.append(mode);cc=c[abs(c[:,2]-mode)<10].copy();cc[:,2]-=mode
        t=fit(cc,aa+mode*scale*n,bb+mode*scale*n,normal,scale,1)
        if t.get('supported') and t['coverage']>=.3:
            t['method']='GX signature inward from repaired physical edge'
            t['search_mode_inset_normalized_px']=40+mode;results.append(t)
    return sorted(results,key=lambda t:t['search_mode_inset_normalized_px'])

def choose_repairs(tracks,candidates,baseline):
    choices=[]
    for side in SIDES:
        prior=tracks['outer'][side];ends=baseline[side]
        suspect=all(c['arc_over_continuation']<.75 for c in ends)
        continuous=all(c['straight_continuation'] is not None and c['straight_continuation']>3 for c in ends)
        plausible=[c for c in candidates[side] if c['offset']>8 and c['corner_score']>=1.25 and c['topology_score']>=.75]
        if suspect and plausible:
            candidate=min(plausible,key=lambda c:c['offset'])
            strong=candidate['corner_score']>10 and candidate['topology_score']>5
            if continuous or strong:
                choices.append(dict(side=side,kind='relocate_physical_edge',candidate=candidate,
                                    reason='both prior corners contradict card termination; earlier rounded candidate supported'))
                continue
        # A bent edge can exceed the old ±14 normalized-pixel curve envelope.
        # Require a large gain in observed support, not merely a more flexible fit.
        # For a same-middle bent edge, retain the original corner identity.
        # A large bend changes the virtual corner used by the continuation
        # probe, so that straight-frame ratio cannot certify the new curve.
        original_corners_supported=all(c['arc_over_continuation']>=.75 for c in ends)
        local=[c for c in candidates[side] if original_corners_supported and abs(c['offset'])<8
               and c['corner_score']>=1.25 and c['track']['coverage']>=prior.get('coverage',0)+.25]
        if local and prior.get('supported'):
            candidate=max(local,key=lambda c:c['track']['coverage'])
            choices.append(dict(side=side,kind='same_edge_curve',candidate=candidate,
                                reason='same middle edge with rounded endpoints and at least a quarter-side more observed support'))
    return choices

def repair(image,geometry,tracks,*,evidence=None):
    result=dict(geometry=copy.deepcopy(geometry),tracks=copy.deepcopy(tracks),repairs=[],review_required=False)
    if evidence is None:
        evidence=dict(hypotheses=hypotheses(image,geometry),baseline=baseline_evidence(image,geometry))
    choices=choose_repairs(tracks,evidence['hypotheses'],evidence['baseline'])
    if not choices:
        from .soft_seam import proposals,certify
        inv=np.linalg.inv(cv2.getPerspectiveTransform(np.float32(geometry['outer']),FRAME))
        lab=None
        for k,side in enumerate(SIDES):
            candidates=proposals(evidence['hypotheses'][side],evidence['baseline'][side])
            # Test the earliest rounded candidate; do not skip a doubtful edge
            # to select a stronger printed line farther inside the card.
            if not candidates:continue
            candidate=min(candidates,key=lambda c:c['offset'])
            if lab is None:lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
            proof=certify(lab,inv,k,candidate,tracks['inner'][side])
            if proof:
                choices.append(dict(side=side,kind='relocate_physical_edge',candidate=candidate,
                                    reason=proof['reason'],soft_seam_evidence=proof))
    result['choices']=choices
    group=same_edge_group(geometry,tracks,choices)
    if group is not None:return group
    # Multiple competing relocations need a jointly verified frame, not greedy repair.
    if len(choices)>1:
        result.update(review_required=True,reason='multiple sides require independent holder repair')
        result['axes']=measure(tracks,geometry['outer']);return result
    if not choices:
        result['axes']=measure(tracks,geometry['outer']);return result
    choice=choices[0];side=choice['side'];k=SIDES.index(side);candidate=choice['candidate']
    H=cv2.getPerspectiveTransform(np.float32(geometry['outer']),FRAME);inv=np.linalg.inv(H)
    rim_keyline=None
    if choice['kind']=='relocate_physical_edge':
        from .outer_rim import choose_rim
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
        candidate,rim_keyline,rim_evidence=choose_rim(lab,inv,k,candidate,evidence['hypotheses'][side])
        if rim_evidence:result['rim_evidence']=rim_evidence
    a,b=FRAME[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
    observed=project(np.concatenate(candidate['track']['segments']),H)
    ts=(observed-a)@v/(v@v);ds=(observed-a)@n
    # A perspective frame from the supported physical edge's straight comparison.
    p=np.polynomial.polynomial.polyfit(ts-.5,ds,1)
    q=FRAME.copy()
    for index,t in [(k,0),((k+1)%4,1)]:q[index]+=n*np.polynomial.polynomial.polyval(t-.5,p)
    result['geometry']['outer']=project(q,inv).tolist()
    result['tracks']['outer'][side]=copy.deepcopy(candidate['track'])
    inner=tracks['inner'][side]
    inner_points=project(np.concatenate(inner['segments']),H) if inner.get('supported') and inner.get('segments') else np.empty((0,2))
    gap=float(np.median((inner_points-a)@n)-candidate['offset']) if len(inner_points) else -1
    inner_action='retained'
    # A correctly selected physical rim may leave the old inner fit on the
    # first dark band. Validate the same sequence without moving that outer.
    if rim_keyline is None and choice['kind']=='relocate_physical_edge' and 5<gap<24:
        expanded=evidence['hypotheses'][side]
        sequences=[]
        for dark in expanded:
            if not 9<=dark['offset']-candidate['offset']<=24:continue
            edge,keyline,proof=choose_rim(lab,inv,k,dark,expanded)
            if keyline is not None and abs(edge['offset']-candidate['offset'])<3:
                sequences.append((keyline,proof))
        if len(sequences)==1:
            rim_keyline,result['rim_evidence']=sequences[0]
        else:
            # A blurred shoulder can lack a separate long outer-gradient fit
            # while its local GX signature is still observed in source pixels.
            signatures=inner_from_physical_edge(image,np.array(result['geometry']['outer']),k)
            if signatures:
                proposed=max(signatures,key=lambda t:t['coverage'])
                pts=project(np.concatenate(proposed['segments']),H)
                inset=float(np.median((pts-a)@n)-candidate['offset'])
                if inset-gap>3:
                    rim_keyline=dict(track=proposed)
                    result['rim_evidence']=dict(reason='first complete local GX signature lies inward of prior dark-band fit',observed_keyline_inset=inset)
    if rim_keyline is not None:
        result['tracks']['inner'][side]=copy.deepcopy(rim_keyline['track'])
        inner_action='keyline independently supported by GX rim/signature evidence'
    elif gap<=5 or choice['kind']=='same_edge_curve':
        seed=np.array(geometry['inner'],float)
        if gap<=5:
            plane=project(seed,H)
            plane[[k,(k+1)%4]]+=n*candidate['offset']
            seed=project(plane,inv)
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32');outer=np.array(result['geometry']['outer'])
        scale=(np.linalg.norm(outer[1]-outer[0])+np.linalg.norm(outer[2]-outer[3]))/2520
        c,aa,bb,normal=extract(lab,seed,k,'inner_colour',scale,allow_partial_profiles=True,search_radius=32)
        alternatives=[fit(c,aa,bb,normal,scale,d) for d in [1,2]]
        usable=[t for t in alternatives if t.get('supported')]
        if gap<=5 and not usable:
            alternatives+=inner_from_physical_edge(image,outer,k)
            usable=[t for t in alternatives if t.get('supported')]
        if gap<=5:
            result['tracks']['inner'][side]=max(usable,key=lambda t:t.get('coverage',0)) if usable else dict(supported=False,reason='relocated outer invalidates old inner; new keyline unsupported')
            inner_action='refitted after invalid outer/inner ordering' if usable else 'withheld'
        elif usable:
            improved=max(usable,key=lambda t:t.get('coverage',0))
            if improved['coverage']>=inner.get('coverage',0)+.25:
                result['tracks']['inner'][side]=improved;inner_action='curved keyline gains observed support'
        result['inner_alternatives']=alternatives
    result['repairs']=[dict(side=side,kind=choice['kind'],offset_normalized_px=candidate['offset'],inner_action=inner_action)]
    result['review_required']=True
    result['axes']=measure(result['tracks'],result['geometry']['outer'])
    return result

def audit_repair(image,geometry,tracks,result):
    """Rerun holder hypotheses at 75% size and JPEG85 with a fixed seed frame.

    Boundary agreement uses the fit residual band, not a centering allowance.
    This audits the holder repair; it does not certify fresh localization.
    """
    if not result.get('repairs'):return dict(stable=True,checks=[],scope='no repair proposed')
    originals=result['repairs'];checks=[]
    from .quality_evidence import disagreement
    for label,im in [('resize75',cv2.resize(image,None,fx=.75,fy=.75,interpolation=cv2.INTER_AREA)),
                     ('jpeg85',cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[1],cv2.IMREAD_COLOR))]:
        factor=np.array([im.shape[1]/image.shape[1],im.shape[0]/image.shape[0]])
        g={k:(np.array(geometry[k])*factor).tolist() for k in ['outer','inner']};prior=copy.deepcopy(tracks)
        for sides in prior.values():
            for t in sides.values():t['segments']=[(np.array(s)*factor).tolist() for s in t.get('segments',[])]
        candidate=repair(im,g,prior);repaired=candidate.get('repairs',[])
        same=bool(len(repaired)==len(originals) and all(any(r['side']==o['side'] and r['kind']==o['kind']
                  and abs(r['offset_normalized_px']-o['offset_normalized_px'])<=3 for r in repaired) for o in originals))
        comparisons={}
        for original in originals:
            side=original['side'];k=SIDES.index(side)
            for kind in ['outer','inner']:
                observed=copy.deepcopy(candidate['tracks'][kind][side])
                observed['segments']=[(np.array(s)/factor).tolist() for s in observed.get('segments',[])]
                a,b=np.asarray(geometry[kind],float)[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
                q=np.asarray(geometry['outer']);scale=(np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2520
                key=kind if len(originals)==1 else side+'_'+kind
                comparisons[key]=disagreement(result['tracks'][kind][side],observed,a,b,n,scale)
        compatible=all(c['common_profiles']>=54 and c.get('p95_normalized_px',float('inf'))<=3 for c in comparisons.values())
        if result.get('same_edge_agreement'):
            # A resize can reduce the support gain enough to retain an already
            # correct curve. For same-edge refinements, compare the actual
            # observed boundaries and reportable spans, not the action count.
            same=bool(all(r['kind']=='same_edge_curve' and r['side'] in {o['side'] for o in originals} for r in repaired)
                      and all(a.get('measured') for a in candidate['axes'].values()))
        checks.append(dict(variant=label,repairs=repaired,same_middle_boundary=same,compatible_observed_tracks=compatible,comparisons=comparisons,axes=candidate['axes']))
    return dict(stable=all(c['same_middle_boundary'] and c['compatible_observed_tracks'] for c in checks),checks=checks,
                scope='holder hypotheses and inner repair rerun; compare common observed tracks; fixed/scaled seed, not localization')
