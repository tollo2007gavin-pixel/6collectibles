"""Diagnostic smooth boundary tracks, independent of centering targets."""
import numpy as np
import cv2
from .yellow_segments import profiles


def track_boundary(img,outer,inner,index,kind,ppm,*,color_space='bgr',smoothing_mm=0.,model_mode='adaptive',trim=.035,transition_mm=.20,along_sigma=1.2):
    if model_mode not in ('adaptive','forced'):
        raise ValueError('model_mode must be adaptive or forced')
    data=profiles(img,outer,inner,index,kind,ppm,trim=trim,n_samples=360,
                  color_space=color_space,smoothing_mm=smoothing_mm,transition_mm=transition_mm,along_sigma=along_sigma)
    if data is None:return None
    t,d,w,rows=(data[k] for k in ('t','offsets','weight','rows'))
    rng=np.random.default_rng(71)
    tolerance=max(.8,.035*ppm)
    strongest=[]
    for row in np.unique(rows):
        choices=np.flatnonzero(rows==row)
        strongest.append(choices[np.argmax(w[choices])])
    strongest=np.asarray(strongest)
    def model_score(model):
        good=np.abs(d-np.polyval(model,t))<tolerance
        votes=np.zeros(360);np.maximum.at(votes,rows[good],w[good])
        chunks=votes.reshape(12,30).sum(axis=1)
        return votes.sum()+15*np.count_nonzero(chunks>=10)+15*((chunks[0]>=10)+(chunks[-1]>=10))
    tracks=[]
    for degree in (1,2,3):
        best=np.polyfit(t[strongest],d[strongest],degree)
        score=model_score(best)
        for _ in range(550):
            ix=rng.choice(len(t),degree+1,replace=False)
            if np.ptp(t[ix])<.65 or min(np.diff(np.sort(t[ix])))<.07:continue
            model=np.polyfit(t[ix],d[ix],degree)
            if np.max(np.abs(np.polyval(model,np.linspace(trim,1-trim,30))))>data['radius']*1.2:continue
            value=model_score(model)
            if value>score:best,score=model,value
        if best is None:continue
        for _ in range(5):
            good=np.abs(d-np.polyval(best,t))<tolerance
            selected=[]
            for row in np.unique(rows[good]):
                choices=np.flatnonzero(good&(rows==row))
                selected.append(choices[np.argmax(w[choices])])
            selected=np.array(selected)
            if len(selected)<50:break
            bins=np.minimum(11,rows[selected]//30)
            counts=np.bincount(bins,minlength=12)
            weight=w[selected]/np.maximum(counts[bins],1)
            best=np.polyfit(t[selected],d[selected],degree,w=np.sqrt(weight))
        if len(selected)<50:continue
        counts=np.bincount(np.minimum(11,rows[selected]//30),minlength=12)
        covered=counts>=10
        residual=d[selected]-np.polyval(best,t[selected])
        # Added degrees must recover substantial distributed support, rather
        # than merely fit noise more tightly on the same edge samples.
        criterion=len(selected)-36*(degree-1)
        tracks.append(dict(model=best,degree=degree,criterion=criterion,
                           support=len(selected)/360,counts=counts.tolist(),
                           covered_segments=int(covered.sum()),
                           rms_px=float(np.sqrt(np.mean(residual**2))),
                           t=t[selected],d=d[selected]))
    if not tracks:return None
    eligible=[track for track in tracks if (model_mode=='adaptive' or track['degree']>1) and track['covered_segments']>=10 and
              min(track['counts'][0],track['counts'][-1])>=10]
    if not eligible:return None
    best=max(eligible,key=lambda x:x['criterion'])
    ts=np.linspace(trim,1-trim,160)
    offsets=np.polyval(best['model'],ts)
    a,b,n=(data[k] for k in ('a','b','inward'))
    points=a+ts[:,None]*(b-a)+offsets[:,None]*n
    observed=a+best['t'][:,None]*(b-a)+best['d'][:,None]*n
    return dict(points=points.tolist(),observations=observed.tolist(),
                degree=best['degree'],coefficients=best['model'].tolist(),
                support=best['support'],covered_segments=best['covered_segments'],
                rms_px=best['rms_px'],counts=best['counts'],
                straight_support=tracks[0]['support'],
                t_range=[trim,1-trim],a=a.tolist(),b=b.tolist(),inward=n.tolist())


def checked_track(img,outer,inner,index,kind,ppm,*,color_space='bgr',smoothing_mm=0.,model_mode='adaptive',trim=.035,transition_mm=.20,along_sigma=1.2):
    """Withhold a track that moves substantially after ordinary resampling."""
    track=track_boundary(img,outer,inner,index,kind,ppm,color_space=color_space,smoothing_mm=smoothing_mm,model_mode=model_mode,trim=trim,transition_mm=transition_mm,along_sigma=along_sigma)
    if track is None:return None
    small=cv2.resize(img,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    scale=np.array([small.shape[1]/img.shape[1],small.shape[0]/img.shape[0]])
    other=track_boundary(small,(outer+.5)*scale-.5,(inner+.5)*scale-.5,
                         index,kind,ppm*float(np.sqrt(np.prod(scale))),color_space=color_space,smoothing_mm=smoothing_mm,model_mode=model_mode,trim=trim,transition_mm=transition_mm,along_sigma=along_sigma)
    delta=None
    if other is not None:
        mapped=(np.asarray(other['points'])+.5)/scale-.5
        delta=float(np.max(np.linalg.norm(mapped-np.asarray(track['points']),axis=1)))
        track['half_resolution_points']=mapped.tolist()
        track['half_resolution_support']=other['support']
    track['resampling_max_shift_px']=delta
    track['resampling_stable']=bool(delta is not None and delta<=2 and other['support']>=.75)
    track['profile_color_space']=color_space
    track['profile_smoothing_mm']=smoothing_mm
    track['profile_transition_mm']=transition_mm
    track['profile_along_sigma']=along_sigma
    return track


def retention_reasons(track,ppm,*,allow_straight=False):
    """A visible curve must improve coverage, remain local, and survive resizing."""
    if track is None:return ['no distributed edge support']
    reasons=[]
    if not allow_straight and track['degree']<=1:reasons.append('straight edge preferred')
    if track['support']<.75:reasons.append('insufficient edge support')
    if track['covered_segments']<10 or min(track['counts'][0],track['counts'][-1])<10:
        reasons.append('insufficient support at ends or across side')
    if not allow_straight and track['support']-track['straight_support']<.08:
        reasons.append('curve does not recover enough additional support')
    if not track['resampling_stable']:reasons.append('unstable after resampling')
    offsets=np.polyval(track['coefficients'],np.linspace(*track['t_range'],160))
    if not np.isfinite(offsets).all() or np.max(np.abs(offsets))>max(3.,.85*ppm)*1.2:
        reasons.append('curve leaves the local edge search')
    return reasons


def smoothing_erodes_outer_edge(primary,alternative,inward):
    """Extra blur must not replace a valid physical edge with an inner shoulder.

    This guard compares the same colour signal at different smoothing scales.
    It is only used after the primary has passed its support/resampling gates.
    A systematic inward movement exceeding half a source pixel may select a
    different shoulder; resampling stability alone does not establish accuracy.
    """
    if primary['profile_color_space']!=alternative['profile_color_space']:return False
    if alternative['profile_smoothing_mm']<=primary['profile_smoothing_mm']:return False
    shift=(np.asarray(alternative['points'])-np.asarray(primary['points']))@inward
    # A change of curvature can improve one end while moving the other end
    # outward. Limit this guard to an almost whole-side inward displacement.
    return bool(np.median(shift)>.5 and np.mean(shift>.5)>=.95)


def tracks_agree(a,b):
    """Compare only the shared observed span, never extrapolated endpoints."""
    low=max(a['t_range'][0],b['t_range'][0])
    high=min(a['t_range'][1],b['t_range'][1])
    ts=np.linspace(low,high,160)
    delta=np.abs(np.polyval(a['coefficients'],ts)-np.polyval(b['coefficients'],ts))
    return bool(np.max(delta)<=1.)


def recover_inner_track(img,outer,inner,index,ppm,*,model_mode):
    """Recover missing interiors using contrast and an unobstructed middle span.

    Printed tabs can overlap a side near its ends. Glare/foil texture can also
    interrupt a strong edge row by row. The same support, locality and resize
    gates still apply; two profile variants must agree within one source pixel.
    Existing accepted paths and exterior paths never enter this fallback.
    """
    proposals=[]
    # Keep the original two searches first. Only after both fail, compare
    # intermediate along-edge smoothing on the central 76% of the side.
    for along,trims,reset in ((1.2,(.035,.09),True),(6.,(.035,.09),True),
                             (1.2,(.12,),True),(3.,(.12,),False),(6.,(.12,),False)):
        if reset:proposals=[]
        for trim in trims:
            for metric in ('bgr','lab_chroma','luma'):
                for smooth in (.035,.07):
                    t=checked_track(img,outer,inner,index,'inner',ppm,
                                    color_space=metric,smoothing_mm=smooth,
                                    model_mode=model_mode,trim=trim,along_sigma=along)
                    if not retention_reasons(t,ppm,allow_straight=True):proposals.append(t)
        eligible=[]
        for t in proposals:
            peers=[p for p in proposals if p is not t and tracks_agree(t,p)]
            if peers:
                t['recovery_agreement_count']=len(peers)
                eligible.append(t)
        if eligible:
            # Prefer the broadest span and least blur, then support. Minimizing
            # resize drift alone can reward a heavily blurred wrong shoulder.
            chosen=min(eligible,key=lambda t:(t['t_range'][0],t['profile_smoothing_mm'],
                                             -t['support'],t['resampling_max_shift_px']))
            chosen['profile_selection']='missing interior recovered by agreeing contrast profiles'
            return chosen
    return None


def recover_outer_track(img,outer,inner,index,ppm,*,model_mode):
    """Require agreeing, resize-stable profiles on a missing physical edge.

    Along-edge smoothing suppresses foil/background texture without widening
    the normal search. End trimming avoids rounded corners; observed coverage
    and all ordinary support thresholds still apply. Never replace a fit that
    has already passed the existing checks.
    """
    proposals=[]
    for along in (1.2,6.):
        for trim in (.035,.09,.12):
            for metric in ('bgr','lab_chroma','luma'):
                for smooth in (.035,.07):
                    t=checked_track(img,outer,inner,index,'outer',ppm,
                        color_space=metric,smoothing_mm=smooth,model_mode=model_mode,
                        trim=trim,along_sigma=along)
                    if not retention_reasons(t,ppm,allow_straight=True):proposals.append(t)
    eligible=[]
    for t in proposals:
        peers=[p for p in proposals if p is not t and tracks_agree(t,p)]
        if peers:
            t['recovery_agreement_count']=len(peers)
            eligible.append(t)
    if not eligible:return None
    chosen=min(eligible,key=lambda t:(t['t_range'][0],t['profile_smoothing_mm'],
                                     -t['support'],t['resampling_max_shift_px']))
    chosen['profile_selection']='missing exterior recovered by agreeing contrast profiles'
    return chosen


def attach_boundary_tracks(img,best,*,model_mode='adaptive',allow_inner_recovery=True):
    """Inspect all eight photo borders after final measurement-frame selection.

    Paths describe observed pixels only; they do not alter the source quads.
    The batch engine separately validates and measures retained paths. The
    legacy allow_inner_recovery flag gates both interior and exterior recovery
    so a rejected physical frame cannot grow extra diagnostic fits.
    """
    if best['diag'].get('capture')!='photo':return
    r,diag=best['result'],best['diag']
    outer,inner,ppm=r['outer'],r['inner'],r['px_per_mm']
    candidates={kind:[] for kind in ('outer','inner')}
    tracks={kind:[] for kind in ('outer','inner')}
    for kind in ('outer','inner'):
        for index in range(4):
            try:
                track=checked_track(img,outer,inner,index,kind,ppm,model_mode=model_mode)
                reasons=retention_reasons(track,ppm,allow_straight=True)
                primary_shift=track.get('resampling_max_shift_px') if track else None
                withheld_profiles=[]
                if reasons or primary_shift is None or primary_shift>max(.5,.035*ppm):
                    alternatives=[]
                    for metric,smooth in [('lab_chroma',.035),('lab_chroma',.07),('bgr',.10)]:
                        fallback=checked_track(img,outer,inner,index,kind,ppm,
                                               color_space=metric,smoothing_mm=smooth,model_mode=model_mode)
                        if retention_reasons(fallback,ppm,allow_straight=True):continue
                        if not reasons and kind=='outer':
                            direction=outer[(index+1)%4]-outer[index]
                            inward=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
                            if smoothing_erodes_outer_edge(track,fallback,inward):
                                withheld_profiles.append(dict(color_space=metric,smoothing_mm=smooth,
                                    reason='extra smoothing moves a supported physical edge inward'))
                                continue
                        # A stable primary is replaced only for a large stability
                        # improvement without sacrificing broad edge support.
                        if not reasons and (fallback['support']<max(.75,track['support']-.15) or
                                fallback['resampling_max_shift_px']>primary_shift*.5):continue
                        alternatives.append(fallback)
                    if alternatives:
                        fallback=min(alternatives,key=lambda t:t['resampling_max_shift_px'])
                        fallback['primary_withheld_reasons']=reasons
                        fallback['primary_resampling_shift_px']=primary_shift
                        fallback['profile_selection']='supported profile with substantially improved resampling stability'
                        track=fallback;reasons=[]
                if reasons and kind=='inner' and allow_inner_recovery:
                    recovered=recover_inner_track(img,outer,inner,index,ppm,model_mode=model_mode)
                    if recovered is not None:
                        recovered['primary_withheld_reasons']=reasons
                        recovered['primary_resampling_shift_px']=primary_shift
                        track=recovered;reasons=[]
                if reasons and kind=='outer' and allow_inner_recovery:
                    recovered=recover_outer_track(img,outer,inner,index,ppm,model_mode=model_mode)
                    if recovered is not None:
                        recovered['primary_withheld_reasons']=reasons
                        recovered['primary_resampling_shift_px']=primary_shift
                        track=recovered;reasons=[]
            except (ValueError,cv2.error,np.linalg.LinAlgError):
                track=None;reasons=['edge fitting failed']
            if track is not None:
                if withheld_profiles:track['withheld_profile_alternatives']=withheld_profiles
                track['retained']=not reasons
                track['withheld_reasons']=reasons
            candidates[kind].append(track)
            tracks[kind].append(track if not reasons else None)
    # Independently detected inner and outer paths must retain a positive
    # yellow strip. Evaluate the same positions along each side; withhold both
    # proposals on conflict, preserving the already validated straight frame.
    for index in range(4):
        intervals=[t['t_range'] for kind in tracks if (t:=tracks[kind][index]) is not None]
        low=max((r[0] for r in intervals),default=.035)
        high=min((r[1] for r in intervals),default=.965)
        ts=np.linspace(low,high,160)[:,None]
        points=[]
        for kind,quad in (('outer',outer),('inner',inner)):
            track=tracks[kind][index]
            if track and track['t_range']==[low,high]:
                points.append(np.asarray(track['points']))
            elif track:
                points.append(np.asarray(track['a'])+ts*(np.asarray(track['b'])-track['a'])+
                              np.polyval(track['coefficients'],ts)*np.asarray(track['inward']))
            else:points.append(quad[index]+ts*(quad[(index+1)%4]-quad[index]))
        direction=outer[(index+1)%4]-outer[index]
        inward=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
        if np.min((points[1]-points[0])@inward)<=max(.5,.025*ppm):
            recovered_outer=tracks['outer'][index]
            if recovered_outer and 'recovery_agreement_count' in recovered_outer:
                recovered_outer['retained']=False
                recovered_outer['withheld_reasons'].append('recovered exterior conflicts with interior')
                tracks['outer'][index]=None
                continue  # Preserve the previously accepted interior, including recovered interiors.
            recovered=tracks['inner'][index]
            if recovered and 'recovery_agreement_count' in recovered:
                recovered['retained']=False
                recovered['withheld_reasons'].append('recovered interior conflicts with exterior')
                tracks['inner'][index]=None
                continue  # A new proposal must not remove an accepted exterior.
            for kind in ('outer','inner'):
                track=tracks[kind][index]
                if track:
                    track['retained']=False
                    track['withheld_reasons'].append('inner and outer paths conflict')
                    tracks[kind][index]=None
    # A fully straight photo with already good quads keeps the established
    # measurements (including A/C); the tracker is primary when it finds bow.
    has_curve=any(t is not None and t['degree']>1 for kind in tracks for t in tracks[kind])
    retained=[]
    for kind in ('outer','inner'):
        diag[kind+'_curve_candidates']=candidates[kind]
        if has_curve and any(track is not None for track in tracks[kind]):
            r[kind+'_boundary_paths']=[track['points'] if track else None for track in tracks[kind]]
            diag[kind+'_curve_tracks']=tracks[kind]
            retained.extend(kind+' '+('top','right','bottom','left')[i]
                            for i,track in enumerate(tracks[kind]) if track is not None)
    if retained:
        diag['review_required']=True
        diag.setdefault('review_reasons',[]).append(
            'Photo margins use supported boundary tracks; physical deformation remains uncalibrated')
        diag['measurement_geometry']='observed_boundary_tracks'
        diag['track_coverage']={kind:[t is not None for t in tracks[kind]] for kind in tracks}
