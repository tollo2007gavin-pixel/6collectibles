"""Evidence guard: does a proposed straight boundary continue through corners?

A physical rounded card edge normally terminates before the extrapolated corner.
A plastic seam can continue through it. Flagging does not identify the material
or find a replacement edge. Thresholds are provisional, not accuracy allowances.
"""

import cv2
import numpy as np


def inspect_extent(image,outer):
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    q=np.asarray(outer,float);scale=np.mean([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])])/1260
    fractions=np.linspace(-.025,1.025,421);offsets=np.arange(-5,5.01,.5)
    result={}
    for k,side in enumerate(['top','right','bottom','left']):
        a,b=q[[k,(k+1)%4]];tangent=(b-a)/np.linalg.norm(b-a);normal=np.array([-tangent[1],tangent[0]])
        points=a+fractions[:,None,None]*(b-a)+offsets[None,:,None]*scale*normal
        valid=((points[:,:,0]>=1)&(points[:,:,0]<image.shape[1]-2)&(points[:,:,1]>=1)&(points[:,:,1]<image.shape[0]-2)).all(1)
        colours=cv2.remap(lab,points[:,:,0].astype(np.float32),points[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
        g=np.linalg.norm(np.gradient(cv2.GaussianBlur(colours,(3,1),0),.5,axis=1),axis=2)
        strength=g[:,(offsets>=-1.5)&(offsets<=1.5)].max(1)
        middle=valid&(fractions>=.15)&(fractions<=.85)
        reference=float(np.median(strength[middle])) if middle.any() else 0
        d=dict(middle_gradient=reference,ends=[])
        for lo,hi in [(0,.018),(.982,1.0)]:
            mask=valid&(fractions>=lo)&(fractions<=hi)
            d['ends'].append(dict(n=int(mask.sum()),gradient=float(np.median(strength[mask])) if mask.any() else None,
                relative=float(np.median(strength[mask])/(reference+1e-6)) if mask.any() else None))
        result[side]=d
    return result


def continued_sides(evidence):
    """Require strong evidence at BOTH ends; one glare patch is insufficient."""
    return [side for side,d in evidence.items() if d['middle_gradient']>=2 and
            all(end['n']>=5 and end['relative'] is not None and end['relative']>.75 for end in d['ends'])]


def apply_outer_extent_guard(image,result):
    """Withhold ratios if an outer side contradicts rounded-card termination.

    All ratios depend on the physical perspective frame, so an unresolved outer
    side invalidates that frame for both axes. Preserve the candidate geometry
    and old values for inspection; never shift it to an unverified alternative.
    """
    geometry=result.get('geometry')
    if not geometry or not geometry.get('outer'):return result
    evidence=inspect_extent(image,geometry['outer']);flagged=continued_sides(evidence)
    updated=dict(result,outer_extent=dict(sides=evidence,flagged_sides=flagged,
        method='source_pixel_corner_termination',calibrated=False))
    if flagged:
        updated['unvalidated_axes']=result.get('axes',{})
        updated['axes']={}
        updated['status']='unresolved'
        updated['reasons']=list(result.get('reasons',[]))+[
            'Outer boundary continues through both rounded-corner regions: '+', '.join(flagged)+
            '. Physical card frame requires review; possible holder seam or competing edge.']
    return updated

