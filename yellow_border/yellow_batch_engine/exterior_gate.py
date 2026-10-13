"""Source-image clipping gate, independent of the number of retained fit lines.

Three unavailable physical exterior sides are a hard fail. Failure to fit an
edge alone is insufficient evidence. This is not a stock-photo classifier.
"""
import copy
import cv2
import numpy as np

SIDE_NAMES = ('top', 'right', 'bottom', 'left')


def _yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (hsv[...,0] >= 12) & (hsv[...,0] <= 42) & (hsv[...,1] >= 55) & (hsv[...,2] >= 60)


def inspect_exterior_background(image, analysis):
    """Return conservative per-image-side evidence for a 3-of-4 hard gate.

    At each canvas side, examine the middle 70%, excluding rounded corners.
    A yellow band that starts at the canvas and ends within 10% of the normal
    image dimension supports a cropped card border. Broad yellow surroundings
    do not satisfy this check. Separately, a fitted exterior whose outward
    probe leaves the canvas lacks enough background to establish a cut edge.
    A supported, safely in-frame exterior track prevents a colour-only claim.
    """
    h,w = image.shape[:2]
    if min(h,w) < 16:
        return dict(fail=False, code=None, unavailable_count=0, unavailable_sides=[],
                    sides={}, reason=None)
    yellow = _yellow(image)
    methods = analysis.get('methods', {})
    seeds = analysis.get('seeds') or [t['seed'] for m in methods.values() for t in m.get('trials', [])]
    sides = {}
    for side,name in enumerate(SIDE_NAMES):
        normal_length = h if side in (0,2) else w
        along_length = w if side in (0,2) else h
        along = np.rint(np.linspace(.15*(along_length-1),.85*(along_length-1),160)).astype(int)
        depths = np.arange(max(4,int(normal_length*.10))+1)
        if side == 0: band = yellow[depths[None,:],along[:,None]]
        elif side == 2: band = yellow[h-1-depths[None,:],along[:,None]]
        elif side == 3: band = yellow[along[:,None],depths[None,:]]
        else: band = yellow[along[:,None],w-1-depths[None,:]]
        # A one-pixel compression fleck is not a reliable transition into art.
        nonyellow = ~band
        sustained = nonyellow[:,:-2] & nonyellow[:,1:-1] & nonyellow[:,2:]
        has_end = sustained.any(axis=1)
        first_end = sustained.argmax(axis=1)
        narrow_band = band[:,:2].all(axis=1) & has_end & (first_end >= 2)
        band_fraction = float(narrow_band.mean())
        safely_visible_track = False
        # Index sides by their physical outward direction, not top_side/text orientation.
        for method in methods.values():
            for index,path in enumerate(method.get('paths',{}).get('outer',[])):
                if path is None or not seeds: continue
                points = np.asarray(path,float)
                direction = points[-1]-points[0]
                normal = np.array([direction[1],-direction[0]])
                if np.linalg.norm(normal) < 1e-9: continue
                normal /= np.linalg.norm(normal)
                facing = (1 if normal[0]>0 else 3) if abs(normal[0])>abs(normal[1]) else (2 if normal[1]>0 else 0)
                if facing != side: continue
                probe = points+3*normal
                visible = (probe[:,0]>=0)&(probe[:,0]<=w-1)&(probe[:,1]>=0)&(probe[:,1]<=h-1)
                # A curve on the yellow/art transition is not an exterior
                # edge, even if it was retained by the fitting stage.
                def sample(offset):
                    p = points+offset*normal
                    return cv2.remap(image,p[:,0].astype(np.float32),p[:,1].astype(np.float32),cv2.INTER_LINEAR)
                transition = _yellow(sample(-3)) & ~_yellow(sample(3))
                if float(visible.mean())>=.85 and float(transition.mean())>=.60:
                    safely_visible_track=True
        near_frame = []
        for seed in seeds:
            q=np.asarray(seed['outer'],float)
            for index in range(4):
                direction=q[(index+1)%4]-q[index]
                if np.linalg.norm(direction)<1e-9: continue
                normal=np.array([direction[1],-direction[0]])/np.linalg.norm(direction)
                facing=(1 if normal[0]>0 else 3) if abs(normal[0])>abs(normal[1]) else (2 if normal[1]>0 else 0)
                if facing!=side: continue
                points=q[index]+np.linspace(.15,.85,160)[:,None]*direction
                probe=points+3*normal
                unavailable=(probe[:,0]<0)|(probe[:,0]>w-1)|(probe[:,1]<0)|(probe[:,1]>h-1)
                near_frame.append(float(unavailable.mean()))
        # Multiple plausible seeds must agree, rather than accepting the most clipped one.
        clipped_fraction=min(near_frame,default=0.)
        evidence=[]
        if not safely_visible_track:
            if band_fraction>=.80: evidence.append('narrow yellow border continues to image boundary')
            if clipped_fraction>=.80: evidence.append('exterior frame lacks a 3-pixel outward background strip')
        sides[name]=dict(unavailable=bool(evidence),state='unavailable' if evidence else ('visible' if safely_visible_track else 'unknown'),evidence=evidence,
            narrow_yellow_band_fraction=band_fraction,geometry_clipped_fraction=clipped_fraction,
            supported_exterior_visible=safely_visible_track)
    missing=[name for name in SIDE_NAMES if sides[name]['unavailable']]
    fail=len(missing)>=3
    return dict(fail=fail, code='insufficient_exterior_background' if fail else None,
                unavailable_count=len(missing),unavailable_sides=missing,sides=sides,
                reason=f'Fail: surrounding background unavailable on {len(missing)}/4 exterior sides ({", ".join(missing)})' if fail else None)


def apply_exterior_gate(result, gate):
    """A hard-failed image cannot emit usable ratios or pass downstream averaging."""
    result=copy.deepcopy(result)
    result['image_gate']=copy.deepcopy(gate)
    if gate['fail']:
        result['status']='Fail'
        for axis,value in result['axes'].items():
            result['axes'][axis]=dict(resolved=False,percentage=None,
                reasons=list(dict.fromkeys([gate['reason'],*value.get('reasons',[])])),
                measurement_source='unresolved')
    return result
