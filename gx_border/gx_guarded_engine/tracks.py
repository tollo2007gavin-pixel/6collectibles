"""GX source-pixel tracks. Experimental thresholds, not calibrated accuracy gates.

The seed identifies the physical edge and charcoal keyline; it is NOT a label.
Source Lab profiles retain the GX dark-band/bright-shoulder/darkening sequence.
Curves and straight fits use identical candidates. Missing spans stay missing.
"""
import cv2
import numpy as np

SIDES = ('top', 'right', 'bottom', 'left')


def project(points, H):
    return cv2.perspectiveTransform(np.float32(points)[None], H)[0]


def extract(lab, quad, side, kind, scale, *, allow_partial_profiles=False, search_radius=12.):
    a, b = np.asarray(quad, float)[[side, (side+1)%4]]
    tangent = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-tangent[1], tangent[0]])
    t = np.linspace(.05, .95, 241)
    # Coordinates are source pixels, sampled at a scale-relative step. No warp.
    offsets = np.arange(-46.-search_radius, 13.+search_radius, .5)
    anchors = a+t[:, None]*(b-a)
    points = anchors[:, None]+offsets[None, :, None]*scale*normal
    colours = cv2.remap(lab, points[:,:,0].astype('float32'), points[:,:,1].astype('float32'), cv2.INTER_LINEAR)
    colours = cv2.GaussianBlur(colours, (3,1), 0)
    g = np.gradient(colours, .5, axis=1)
    candidates = []
    for row in range(len(t)):
        valid=(points[row,:,0]>=0)&(points[row,:,0]<lab.shape[1]-1)&(points[row,:,1]>=0)&(points[row,:,1]<lab.shape[0]-1)
        if not allow_partial_profiles and not valid.all():
            continue
        luma = colours[row,:,0]; signed = g[row,:,0]
        strength = np.linalg.norm(g[row], axis=1)
        if kind in ('inner','inner_colour'):
            # The intended keyline is the darkening AFTER the silver shoulder,
            # not whichever art or text transition has the largest gradient.
            neg = np.flatnonzero((signed[1:-1]<=signed[:-2]) & (signed[1:-1]<=signed[2:]) & (signed[1:-1]<-1.5))+1
            pos = np.flatnonzero((signed[1:-1]>=signed[:-2]) & (signed[1:-1]>=signed[2:]) & (signed[1:-1]>1.5))+1
            peaks = []
            endings=neg
            if kind=='inner_colour':
                endings=np.flatnonzero((strength[1:-1]>=strength[:-2])&(strength[1:-1]>=strength[2:])&(strength[1:-1]>2))+1
            for c in endings[np.abs(offsets[endings])<=search_radius]:
                for bidx in pos[(offsets[c]-offsets[pos]>=7)&(offsets[c]-offsets[pos]<=23)]:
                    if kind=='inner_colour' and np.any((endings<c)&(offsets[endings]-offsets[bidx]>=7)):continue
                    for aidx in neg[(offsets[bidx]-offsets[neg]>=9)&(offsets[bidx]-offsets[neg]<=24)]:
                        if aidx<8 or bidx+8>=len(luma): continue
                        if not valid[max(0,aidx-10):max(c+3,bidx+11)].all():continue
                        if min(luma[aidx-8],luma[bidx+8])-np.mean(luma[aidx+4:bidx-2]) >= 10:
                            peaks.append(c)
            peaks = np.unique(peaks).astype(int)
            score = strength if kind=='inner_colour' else -signed
        else:
            peaks = np.flatnonzero((strength[1:-1]>=strength[:-2]) & (strength[1:-1]>=strength[2:]) & (strength[1:-1]>2))+1
            peaks = peaks[np.abs(offsets[peaks])<=search_radius]
            score = strength
        for p in peaks:
            # Gaussian smoothing plus the finite-difference gradient needs
            # two source samples on each side. Padding is never evidence.
            if p<2 or p+2>=len(valid) or not valid[p-2:p+3].all():continue
            u,v,w=score[p-1:p+2]
            delta=np.clip(.5*(u-w)/(u-2*v+w),-.5,.5) if abs(u-2*v+w)>1e-6 else 0
            candidates.append((row,t[row],offsets[p]+.5*delta,float(score[p])**.25))
    return np.asarray(candidates), a, b, normal


def fit(candidates, a, b, normal, scale, degree):
    if len(candidates)<20: return dict(supported=False, reason='too few GX boundary profiles')
    rows,t,d,weight=candidates.T
    rng=np.random.default_rng(712)
    best=None
    for _ in range(350):
        ix=rng.choice(len(t),degree+1,replace=False)
        if np.ptp(t[ix])<.3 or len(np.unique(t[ix]))<=degree:continue
        p=np.polynomial.polynomial.polyfit(t[ix]-.5,d[ix],degree)
        predicted=np.polynomial.polynomial.polyval(t-.5,p)
        if max(abs(np.polynomial.polynomial.polyval(np.array([-.45,0,.45]),p)))>14:continue
        mask=abs(d-predicted)<1.5
        votes=np.zeros(241);np.maximum.at(votes,rows[mask].astype(int),weight[mask])
        value=votes.sum()
        if best is None or value>best[0]: best=value,p
    if best is None:return dict(supported=False,reason='no bounded boundary fit')
    p=best[1]
    for _ in range(3):
        mask=abs(d-np.polynomial.polynomial.polyval(t-.5,p))<1.5
        ix=[]
        for row in np.unique(rows[mask]):
            choices=np.flatnonzero(mask&(rows==row));ix.append(choices[np.argmax(weight[choices])])
        ix=np.array(ix,int)
        if len(ix)<20:return dict(supported=False,reason='insufficient unique profile support')
        p=np.polynomial.polynomial.polyfit(t[ix]-.5,d[ix],degree,w=np.sqrt(weight[ix]))
    order=np.argsort(rows[ix]);ix=ix[order]
    groups=np.split(ix,np.flatnonzero(np.diff(rows[ix])>3)+1)
    groups=[group for group in groups if len(group)>=4]
    segments=[]
    for group in groups:
        ts=np.linspace(t[group[0]],t[group[-1]],max(2,len(group)*2))
        ds=np.polynomial.polynomial.polyval(ts-.5,p)*scale
        segments.append((a+ts[:,None]*(b-a)+ds[:,None]*normal).tolist())
    coverage=sum(t[g[-1]]-t[g[0]] for g in groups)
    return dict(supported=bool(coverage>=.20), degree=degree, support=len(ix)/241,
                coverage=float(coverage), segments=segments, coefficients=p.tolist(),
                rms_normalized_px=float(np.sqrt(np.mean((d[ix]-np.polynomial.polynomial.polyval(t[ix]-.5,p))**2))),
                reason='' if coverage>=.20 else 'insufficient observed span')


def rebuild_frame(tracks, original):
    lines=[]
    for side in SIDES:
        track=tracks['outer'][side]
        if not track.get('supported') or track.get('coverage',0)<.55:return None
        points=np.concatenate(track['segments']);centre=points.mean(0)
        _,_,v=np.linalg.svd(points-centre,full_matrices=False)
        n=v[-1];lines.append(np.r_[n,-n@centre])
    q=[]
    for k in range(4):
        p=np.cross(lines[k-1],lines[k])
        if abs(p[2])<1e-8:return None
        q.append(p[:2]/p[2])
    q=np.float32(q)
    if not cv2.isContourConvex(q):return None
    short=np.linalg.norm(np.asarray(original)[1]-original[0])
    if np.max(np.linalg.norm(q-original,axis=1))>short*.015:return None
    return q


def measure(tracks, outer):
    H=cv2.getPerspectiveTransform(np.float32(outer),np.float32([[0,0],[63,0],[63,88],[0,88]]))
    result={}
    for axis,sides,along,length in [('lr',('left','right'),1,88),('tb',('top','bottom'),0,63)]:
        grid=np.linspace(.05*length,.95*length,721);values=[];masks=[]
        for side in sides:
            for kind in ('outer','inner'):
                track=tracks[kind][side];v=np.full(len(grid),np.nan)
                if track.get('supported'):
                    for segment in track.get('segments',[]):
                        p=project(segment,H);p=p[np.argsort(p[:,along])]
                        if np.any(np.diff(p[:,along])<=0):continue
                        inside=(grid>=p[0,along])&(grid<=p[-1,along])
                        v[inside]=np.interp(grid[inside],p[:,along],p[:,1-along])
                values.append(v);masks.append(np.isfinite(v))
        common=np.all(masks,axis=0)
        widths=np.array([values[1]-values[0],values[2]-values[3]])
        common &= np.all(widths>0,axis=0)
        # Provisional span floor; no yellow perspective/centering allowance.
        if common.mean()<.25:
            result[axis]=dict(measured=False,reason='insufficient common observed span',common_fraction=float(common.mean()))
            continue
        means=widths[:,common].mean(axis=1)
        groups=np.split(np.flatnonzero(common),np.flatnonzero(np.diff(np.flatnonzero(common))>1)+1)
        result[axis]=dict(measured=True,first_pct=float(100*means[0]/sum(means)),
            margins_mm=means.tolist(),common_fraction=float(common.mean()),
            observed_intervals_mm=[[float(grid[g[0]]),float(grid[g[-1]])] for g in groups if len(g)])
    return result


def compare_tracks(image, geometry):
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype('float32')
    q=np.asarray(geometry['outer']);scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    methods={}
    for degree,label in [(1,'source_straight'),(2,'source_curve')]:
        tracks={kind:{} for kind in ('outer','inner')}
        for kind in tracks:
            for k,side in enumerate(SIDES):
                candidates,a,b,normal=extract(lab,geometry[kind],k,kind,scale)
                tracks[kind][side]=fit(candidates,a,b,normal,scale,degree)
        rebuilt=rebuild_frame(tracks,q)
        methods[label]=dict(tracks=tracks,initial_frame=measure(tracks,q),
            rebuilt_frame=measure(tracks,rebuilt) if rebuilt is not None else None,
            rebuilt_outer=rebuilt.tolist() if rebuilt is not None else None)
    return methods
