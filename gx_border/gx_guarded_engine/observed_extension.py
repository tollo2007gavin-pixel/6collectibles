"""Re-observe interrupted boundaries without changing their fitted geometry.

Strong GX signatures establish inner-boundary identity. Between nearby supported
fragments, the boundary itself may remain visible even when foil hides part of
the three-transition signature. Only actual gradient observations can extend
support; unobserved gaps and unseen corners remain empty.
"""
import copy,cv2,numpy as np
from .tracks import extract,SIDES,measure
from .keyline_selection import sampled_track


def extend_track(lab,geometry,kind,side,track):
    if not track.get('supported') or not track.get('segments'):return track,None
    k=SIDES.index(side);q=np.asarray(geometry['outer'])
    scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    a,b=np.asarray(geometry[kind])[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
    points=np.concatenate(track['segments']);ts=(points-a)@v/(v@v);ds=(points-a)@n/scale
    degree=min(2,track.get('degree',2));p=np.polynomial.polynomial.polyfit(ts-.5,ds,degree)
    if np.max(abs(ds-np.polynomial.polynomial.polyval(ts-.5,p)))>.05:return track,None
    # Same gradient evidence as an outer edge, but inner identity is inherited
    # only locally from existing signature-supported fragments on BOTH sides.
    c,_,_,_=extract(lab,geometry[kind],k,'outer',scale,allow_partial_profiles=True)
    if len(c)<20:return track,None
    grid,observed=sampled_track(track,a,b,n,scale);anchor=np.isfinite(observed)
    offsets=np.polynomial.polynomial.polyval(grid-.5,p);rows=[]
    for row in range(len(grid)):
        if anchor[row]:continue
        before=np.flatnonzero(anchor&(grid<grid[row]));after=np.flatnonzero(anchor&(grid>grid[row]))
        if not len(before) or not len(after):continue
        cc=c[c[:,0]==row]
        close=cc[abs(cc[:,2]-offsets[row])<1.5]
        if len(close)!=1:continue
        rows.append(row)
    if not rows:return track,None
    # Longer interrupted signatures need a near-continuous observed edge
    # across the gap, not merely agreement at its two endpoints.
    candidates=set(rows);retained=[]
    missing=np.flatnonzero(~anchor & (grid>grid[anchor].min()) & (grid<grid[anchor].max()))
    for gap in np.split(missing,np.flatnonzero(np.diff(missing)>1)+1):
        if not len(gap):continue
        seen=[i for i in gap if i in candidates]
        if not seen:continue
        distance=grid[gap[-1]]-grid[gap[0]]
        unseen=np.array([i for i in gap if i not in candidates],int)
        runs=np.split(unseen,np.flatnonzero(np.diff(unseen)>1)+1)
        continuous=len(seen)/len(gap)>=.8 and max(map(len,runs))<=3
        if distance<=.12 or continuous:retained.extend(seen)
    rows=sorted(retained)
    if not rows:return track,None
    groups=np.split(np.asarray(rows),np.flatnonzero(np.diff(rows)>1)+1)
    groups=[g for g in groups if len(g)>=4]
    if not groups:return track,None
    result=copy.deepcopy(track);added=[]
    for g in groups:
        t=grid[g];d=np.polynomial.polynomial.polyval(t-.5,p)*scale
        segment=(a+t[:,None]*v+d[:,None]*n).tolist()
        result['segments'].append(segment);added.append([float(t[0]),float(t[-1])])
    gain=sum(hi-lo for lo,hi in added)
    result['coverage']=float(track.get('coverage',0)+gain)
    result['support_extension']=dict(method='observed unique gradient between nearby signature-supported fragments',
                                     added_intervals=added,added_span=gain,fit_unchanged=True)
    return result,result['support_extension']


def extend_incomplete_axes(image,geometry,tracks):
    result=copy.deepcopy(tracks);changes=[];axes=measure(tracks,geometry['outer'])
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
    for axis,sides in [('lr',['left','right']),('tb',['top','bottom'])]:
        if axes[axis].get('measured'):continue
        if any(not tracks[kind][side].get('supported') for side in sides for kind in ['outer','inner']):continue
        for side in sides:
            for kind in ['outer','inner']:
                result[kind][side],proof=extend_track(lab,geometry,kind,side,tracks[kind][side])
                if proof:changes.append(dict(axis=axis,kind=kind,side=side,**proof))
    return result,changes


def audit_extension(image,geometry,prior,result,changes):
    """Repeat the source observations; require reportable common span each time."""
    axes=sorted({c['axis'] for c in changes});original=measure(result,geometry['outer']);checks=[]
    for label,im in [('resize75',cv2.resize(image,None,fx=.75,fy=.75,interpolation=cv2.INTER_AREA)),
                     ('jpeg85',cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[1],cv2.IMREAD_COLOR))]:
        f=np.array([im.shape[1]/image.shape[1],im.shape[0]/image.shape[0]])
        g={k:(np.asarray(geometry[k])*f).tolist() for k in ['outer','inner']};scaled=copy.deepcopy(prior)
        for kinds in scaled.values():
            for t in kinds.values():t['segments']=[(np.asarray(s)*f).tolist() for s in t.get('segments',[])]
        recovered,proof=extend_incomplete_axes(im,g,scaled);values=measure(recovered,g['outer'])
        for axis in axes:
            gain=sum(c['added_span'] for c in changes if c['axis']==axis)
            repeated=sum(c['added_span'] for c in proof if c['axis']==axis)
            measured=values[axis].get('measured',False)
            checks.append(dict(variant=label,axis=axis,stable=bool(measured and repeated>=.5*gain),
                repeated_added_span=repeated,original_added_span=gain,axis_result=values[axis],
                signed_shift_pp=values[axis]['first_pct']-original[axis]['first_pct'] if measured and original[axis].get('measured') else None))
    return dict(checks=checks,stable=all(c['stable'] for c in checks),scope='source observations rerun; original fitted geometry held fixed')
