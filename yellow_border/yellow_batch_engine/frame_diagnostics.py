"""Advisory frame candidates and axis-local measurement sensitivity.

This module never changes acceptance. Callers can explicitly select a candidate.
Diagnostic percentages
are not confidence intervals. In particular, agreement of related fits is not
independent evidence of accuracy.
"""
import numpy as np
import cv2
from .cardcv.photo_margins import boundary_margins

ERRORS=(ValueError,RuntimeError,cv2.error,np.linalg.LinAlgError)
AXES={'lr':((3,1),'left_pct'), 'tb':((0,2),'top_pct')}


def _points(track,key):
    if not track or not track.get('retained') or track.get(key) is None:
        raise ValueError('Required retained boundary evidence is unavailable')
    points=np.asarray(track[key],float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('Invalid boundary points')
    return points


def _line(points):
    center=points.mean(axis=0)
    _,singular,vt=np.linalg.svd(points-center,full_matrices=False)
    if singular[0]<1e-6:raise ValueError('Degenerate boundary span')
    normal=np.array([-vt[0,1],vt[0,0]])
    return normal,float(-normal@center)


def refit_outer_frame(seed,tracks,*,half=False):
    """Intersect straight supporting lines of four retained final outer paths."""
    key='half_resolution_points' if half else 'points'
    lines=[_line(_points(t,key)) for t in tracks['outer']]
    quad=[]
    for i in range(4):
        a=np.array([lines[i-1][0],lines[i][0]])
        if abs(np.linalg.det(a))<1e-6:raise ValueError('Adjacent supporting lines are nearly parallel')
        quad.append(np.linalg.solve(a,-np.array([lines[i-1][1],lines[i][1]])))
    quad=np.asarray(quad);original=np.asarray(seed['outer'],float)
    if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
        raise ValueError('Rebuilt frame is not finite and convex')
    if np.sign(cv2.contourArea(quad.astype(np.float32),oriented=True))!=np.sign(cv2.contourArea(original.astype(np.float32),oriented=True)):
        raise ValueError('Rebuilt frame changes orientation')
    # Locality protects the candidate construction; this is not a new image gate.
    if not np.isfinite(seed['ppm']) or seed['ppm']<=0:
        raise ValueError('Invalid frame scale')
    if np.max(np.linalg.norm(quad-original,axis=1))>2*seed['ppm']:
        raise ValueError('Rebuilt corners exceed the 2 mm candidate locality bound')
    return quad


def _measure(seed,tracks,quad,axis,*,half=False):
    sides,key=AXES[axis];indices=[(s+seed['top_side'])%4 for s in sides]
    point_key='half_resolution_points' if half else 'points'
    paths={k:[None]*4 for k in ('outer','inner')}
    for kind in paths:
        for i in indices:paths[kind][i]=_points(tracks[kind][i],point_key).tolist()
    measured=boundary_margins(quad,seed['inner'],paths['outer'],paths['inner'],seed['top_side'],axes=(axis,))
    return dict(percentage=measured['ratios'][key],nominal_margins_mm=measured['margins_mm'],sampling=measured['axes'][axis])


def _pixel_widths(seed,tracks,quad,axis):
    """P10 of observed border widths in source pixels at rectified cross-sections."""
    sides,_=AXES[axis];along=1 if axis=='lr' else 0;length=88. if along else 63.
    target=np.float32([[0,0],[63,0],[63,88],[0,88]])
    matrix=cv2.getPerspectiveTransform(np.roll(quad,-seed['top_side'],axis=0).astype(np.float32),target)
    inverse=np.linalg.inv(matrix);rect={};bounds=[]
    def transform(p,h):
        points=np.column_stack([p,np.ones(len(p))])@h.T
        if np.any(abs(points[:,2])<1e-9):raise ValueError('Invalid projected points')
        return points[:,:2]/points[:,2:]
    for kind in ('outer','inner'):
        for side in sides:
            p=transform(_points(tracks[kind][(side+seed['top_side'])%4],'points'),matrix)
            rect[kind,side]=p;bounds.append((p[:,along].min(),p[:,along].max()))
    low=max(.1*length,max(a for a,b in bounds));high=min(.9*length,min(b for a,b in bounds))
    positions=np.linspace(low,high,401);widths=[]
    for side in sides:
        mapped=[]
        for kind in ('outer','inner'):
            p=rect[kind,side];order=np.argsort(p[:,along]);cross=np.interp(positions,p[order,along],p[order,1-along])
            q=np.column_stack([cross,positions]) if along else np.column_stack([positions,cross])
            mapped.append(transform(q,inverse))
        widths.append(float(np.percentile(np.linalg.norm(mapped[0]-mapped[1],axis=1),10)))
    return widths


def evaluate_frames(seed,tracks):
    """Keep rejected/missing observations diagnostic; no straight fallback used."""
    result=dict(version=1,active_frame='initial',advisory_only=True,axes={})
    initial=np.asarray(seed['outer'],float);candidate=half_candidate=None
    try:
        candidate=refit_outer_frame(seed,tracks)
        result['candidate_frame']=dict(available=True,quad=candidate.tolist(),
            max_corner_shift_px=float(np.max(np.linalg.norm(candidate-initial,axis=1))))
    except ERRORS as error:result['candidate_frame']=dict(available=False,reason=str(error))
    try:half_candidate=refit_outer_frame(seed,tracks,half=True)
    except ERRORS:pass
    for axis in AXES:
        r=dict(candidate_available=False,advisory_only=True)
        result['axes'][axis]=r
        try:base=_measure(seed,tracks,initial,axis)
        except ERRORS as error:
            r['reason']=str(error);continue
        r['initial_percentage']=base['percentage']
        try:r['initial_resampling_shift_pp']=_measure(seed,tracks,initial,axis,half=True)['percentage']-base['percentage']
        except ERRORS:pass
        if candidate is None:
            r['reason']=result['candidate_frame']['reason'];continue
        try:
            fitted=_measure(seed,tracks,candidate,axis);r['candidate_available']=True
            r['candidate_percentage']=fitted['percentage'];r['candidate_nominal_margins_mm']=fitted['nominal_margins_mm']
            r['frame_difference_pp']=fitted['percentage']-base['percentage']
            r['sampled_fraction']=fitted['sampling']['sampled_fraction']
            r['source_border_width_px_p10']=_pixel_widths(seed,tracks,candidate,axis)
        except ERRORS as error:
            r['reason']=str(error);continue
        # Three separate perturbations distinguish paths, frame, and their interaction.
        try:r['path_resampling_shift_pp']=_measure(seed,tracks,candidate,axis,half=True)['percentage']-fitted['percentage']
        except ERRORS:pass
        if half_candidate is not None:
            try:r['frame_resampling_shift_pp']=_measure(seed,tracks,half_candidate,axis)['percentage']-fitted['percentage']
            except ERRORS:pass
            try:r['candidate_resampling_shift_pp']=_measure(seed,tracks,half_candidate,axis,half=True)['percentage']-fitted['percentage']
            except ERRORS:pass
    return result


def combine_frame_diagnostics(method_axes):
    """Use the active methods' provenance; keep unqualified evidence labeled."""
    qualified={m:a for m,a in method_axes.items() if a['resolved']}
    contributing=qualified or method_axes
    evidence={m:a.get('frame_diagnostics',{}) for m,a in contributing.items()}
    if not any(evidence.values()):return None
    available=bool(evidence) and all(d.get('candidate_available') for d in evidence.values())
    active_frames={d.get('active_frame','initial') for d in evidence.values()}
    result=dict(active_frame=next(iter(active_frames)) if len(active_frames)==1 else 'mixed',advisory_only=True,baseline_qualified=bool(qualified),candidate_available=available,
                contributing_methods=list(contributing),method_diagnostics=evidence)
    if available:
        result['candidate_percentage']=float(np.mean([d['candidate_percentage'] for d in evidence.values()]))
        result['frame_difference_pp']=float(np.mean([d['frame_difference_pp'] for d in evidence.values()]))
    for field in ('initial_resampling_shift_pp','candidate_resampling_shift_pp','path_resampling_shift_pp','frame_resampling_shift_pp'):
        # A missing perturbation check is unavailable, never zero/"stable".
        if all(field in d for d in evidence.values()):result['max_abs_'+field]=max(abs(d[field]) for d in evidence.values())
    return result
