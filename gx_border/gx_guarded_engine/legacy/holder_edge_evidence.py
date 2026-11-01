"""Local median profiles for holder-side diagnostics."""
import cv2,numpy as np

def profiles(image,c,side,edge,*,half_window=25):
    H=cv2.getPerspectiveTransform(np.float32(c['outer']),np.float32([[100,100],[1359,100],[1359,1859],[100,1859]]))
    rect=cv2.warpPerspective(image,H,(1460,1960));lab=cv2.cvtColor(rect,cv2.COLOR_BGR2LAB).astype(np.float32)
    q=cv2.perspectiveTransform(np.float32(c[edge])[None],H)[0]
    vertical=side in ('left','right');coord,axis=(0,1) if vertical else (1,0)
    length=1760 if vertical else 1260
    ids={'left':[0,3],'right':[1,2],'top':[0,1],'bottom':[3,2]}[side]
    if not vertical:lab=lab.transpose(1,0,2)
    if side in ('right','bottom'):
        lab=lab[:,::-1].copy();q[:,coord]=lab.shape[1]-1-q[:,coord]
    prior=np.polyfit(q[ids,axis],q[ids,coord],1);records=[]
    # Independent local medians suppress foil texture without mixing distant sides.
    for t in np.linspace(100+.1*length,100+.9*length,17):
        p=np.median(lab[round(t)-half_window:round(t)+half_window+1],axis=0)
        p=cv2.GaussianBlur(p[None],(5,1),0)[0];g=np.gradient(p,axis=0)
        expected=np.polyval(prior,t);lo=round(expected)-10;hi=round(expected)+11
        strengths={'lab':np.linalg.norm(g,axis=1),'chroma':np.linalg.norm(g[:,1:],axis=1),'luma':np.abs(g[:,0])}
        record=dict(t=float(t),expected=float(expected),peaks={})
        for mode,gg in strengths.items():
            peaks=np.flatnonzero((gg[lo:hi]>=gg[lo-1:hi-1])&(gg[lo:hi]>=gg[lo+1:hi+1])&(gg[lo:hi]>.5))+lo
            row=[]
            for peak in peaks:
                a,b,z=gg[peak-1:peak+2];d=.5*(a-z)/(a-2*b+z) if abs(a-2*b+z)>1e-6 else 0
                row.append([float(peak+np.clip(d,-.5,.5)-expected),float(gg[peak])])
            record['peaks'][mode]=row
        records.append(record)
    return records,H,q,ids,coord,axis,lab.shape[1],prior
