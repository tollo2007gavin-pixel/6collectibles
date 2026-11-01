"""Weak transparent seams: require rounded endpoints AND a physical GX rim.

This supplements continuation-based holder evidence. A printed dark band cannot
qualify merely because it has sharper corners than the physical card edge.
"""
import numpy as np
from .holder_topology import FRAME,sample
from .outer_rim import profile
from .tracks import project


def proposals(candidates,baseline):
    return [c for c in candidates if c['offset']>8 and c['track']['coverage']>=.75
            and c['topology_score']>=1.5
            and all(new['score']>1.75*old['score'] for old,new in zip(baseline,c['corners']))
            # Preserve the weak-seam route. When JPEG/foil contrast puts a
            # corner above its brightness cutoff, require stronger relative
            # rounded-termination evidence at BOTH endpoints instead.
            and (all(old['score']<3.5 for old in baseline)
                 or all(new['arc_over_continuation']>1.5*old['arc_over_continuation']
                        for old,new in zip(baseline,c['corners'])))]


def certify(lab,inv,k,candidate,inner,*,diagnostic=False):
    """Test four transitions against the already observed inner keyline.

No expected centering ratio is used. Missing samples and off-image padding do
not contribute. The positive rim must precede dark band, bright shoulder and
the supported inner keyline, in that order on both halves of the side.
"""
    if not inner.get('supported'):return None
    p=profile(lab,inv,k,candidate);t=p['t'];a,b=FRAME[[k,(k+1)%4]];v=b-a
    n=np.array([-v[1],v[0]])/np.linalg.norm(v);key=np.full(len(t),np.nan)
    for segment in inner.get('segments',[]):
        points=project(segment,np.linalg.inv(inv));ts=(points-a)@v/(v@v);ds=(points-a)@n
        order=np.argsort(ts);ts,ds=ts[order],ds[order]
        if len(ts)<2 or np.any(np.diff(ts)<=0):continue
        mask=(t>=ts[0])&(t<=ts[-1]);key[mask]=np.interp(t[mask],ts,ds)
    common=p['valid']&np.isfinite(key)
    if common.sum()<54:return None
    offsets=np.arange(-4.,81.,.5)
    colors,valid=sample(lab,a+t[:,None,None]*v+(p['d'][:,None,None]+offsets[None,:,None])*n,inv)
    luma=colors[:,:,0];grad=np.gradient(luma,.5,axis=1);votes=np.zeros(len(t),bool)
    for row in np.flatnonzero(common & (p['delta']>4)):
        g=grad[row];z=key[row]-p['d'][row]
        neg=np.flatnonzero((g[1:-1]<=g[:-2])&(g[1:-1]<=g[2:])&(g[1:-1]<-1.5))+1
        pos=np.flatnonzero((g[1:-1]>=g[:-2])&(g[1:-1]>=g[2:])&(g[1:-1]>1.5))+1
        for dark in neg[(offsets[neg]>=9)&(offsets[neg]<=26)]:
            for shoulder in pos[(offsets[pos]-offsets[dark]>=9)&(offsets[pos]-offsets[dark]<=24)]:
                if not 7<=z-offsets[shoulder]<=23:continue
                end=int(np.searchsorted(offsets,z+2))
                if end>=len(offsets) or not valid[row,:end+1].all():continue
                if min(luma[row,dark-6],luma[row,shoulder+6])-np.mean(luma[row,dark+3:shoulder-2])<10:continue
                votes[row]=True
    count=int(votes.sum());fraction=float(votes[common].mean())
    halves=[int(np.sum(votes&(t<.5))),int(np.sum(votes&(t>=.5)))]
    # Foil can interrupt the dark-band sequence. Require a supermajority of
    # supported keyline positions, with independent observations on both halves.
    if count<54 or fraction<2/3 or min(halves)<12 or np.mean(p['delta'][common]>4)<.8:
        return dict(common=int(common.sum()),votes=count,fraction=fraction,halves=halves) if diagnostic else None
    return dict(common_profiles=int(common.sum()),sequence_profiles=count,sequence_fraction=fraction,half_counts=halves,
                reason='weak seam replaced only by rounded bright rim preceding observed GX band, shoulder and keyline')
