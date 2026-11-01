import cv2,numpy as np
from .gx_before import _ransac_keyline, _order_corners, _intersect_lines


def quad_from_models(models,w,h):
 al,bl=models['left'];ar,br=models['right'];at,bt=models['top'];ab,bb=models['bottom']
 lines={'left':(1.,-al,-bl),'right':(1.,ar,-(w-1-br)),'top':(-at,1.,-bt),'bottom':(ab,1.,-(h-1-bb))}
 def meet(v,h):
  a,b,c=lines[v];d,e,f=lines[h];return np.linalg.solve([[a,b],[d,e]],[-c,-f])
 return np.float32([meet('left','top'),meet('right','top'),meet('right','bottom'),meet('left','bottom')])

def stripe_frame(bgr,seed):
 w,h=1260,1760; pad=110
 dst=np.float32([[pad,pad],[pad+w-1,pad],[pad+w-1,pad+h-1],[pad,pad+h-1]])
 H=cv2.getPerspectiveTransform(np.float32(seed),dst)
 rect=cv2.warpPerspective(bgr,H,(w+2*pad,h+2*pad));gray=cv2.cvtColor(rect,cv2.COLOR_BGR2GRAY)
 models={};diagnostics={}
 for side in ['left','right','top','bottom']:
  vertical=side in ['left','right'];dim,length=(gray.shape[1],gray.shape[0]) if vertical else gray.shape
  fractions=np.r_[np.linspace(.12,.30,90),np.linspace(.70,.88,90)] if vertical else np.linspace(.12,.88,180)
  along=(pad+fractions*(length-2*pad)).astype(int)
  profiles=np.array([gray[t,:] if vertical else gray[:,t] for t in along],np.float32)
  if side in ['right','bottom']:profiles=profiles[:,::-1].copy()
  profiles=cv2.GaussianBlur(profiles,(5,1),0)
  g=np.gradient(profiles,axis=1);cand=[]
  for j,t in enumerate(along):
   lo,hi=pad-45,pad+85
   neg=np.where((g[j,lo+1:hi-1]<=g[j,lo:hi-2])&(g[j,lo+1:hi-1]<=g[j,lo+2:hi])&(g[j,lo+1:hi-1]<-1))[0]+lo+1
   pos=np.where((g[j,lo+1:hi-1]>=g[j,lo:hi-2])&(g[j,lo+1:hi-1]>=g[j,lo+2:hi])&(g[j,lo+1:hi-1]>1))[0]+lo+1
   for a in neg:
    for b in pos[(pos-a>=10)&(pos-a<=22)]:
     strength=min(-g[j,a],g[j,b])
     cand.append((t,.5*(a+b),strength))
  m,s=_ransac_keyline(np.array(cand),pad+27,35,tol_px=3,iters=1200)
  models[side]=m;diagnostics[side]=s
 stripe=quad_from_models(models,*gray.shape[::-1]);source=cv2.perspectiveTransform(stripe[None],np.linalg.inv(H))[0]
 return source,diagnostics

def fit_frame(bgr,seed,stripe=None):
 if stripe is None: stripe,diagnostics=stripe_frame(bgr,seed)
 w,h=1260,1760;pad=100
 dst=np.float32([[pad,pad],[pad+w-1,pad],[pad+w-1,pad+h-1],[pad,pad+h-1]])
 H=cv2.getPerspectiveTransform(stripe,dst);rect=cv2.warpPerspective(bgr,H,(w+2*pad,h+2*pad))
 gray=cv2.cvtColor(rect,cv2.COLOR_BGR2GRAY).astype(np.float32);lab=cv2.cvtColor(rect,cv2.COLOR_BGR2LAB).astype(np.float32)
 outer={};inner={};dbg={}
 for side in ['left','right','top','bottom']:
  vertical=side in ['left','right'];dim,length=(gray.shape[1],gray.shape[0]) if vertical else gray.shape
  fractions=np.r_[np.linspace(.12,.30,90),np.linspace(.70,.88,90)] if vertical else np.linspace(.12,.88,180)
  along=(pad+fractions*(length-2*pad)).astype(int)
  profiles=np.array([gray[t,:] if vertical else gray[:,t] for t in along],np.float32)
  colour=np.array([lab[t,:] if vertical else lab[:,t] for t in along],np.float32)
  if side in ['right','bottom']:profiles=profiles[:,::-1].copy();colour=colour[:,::-1].copy()
  profiles=cv2.GaussianBlur(profiles,(3,1),0);colour=cv2.GaussianBlur(colour,(3,1),0)
  # Gradient magnitudes on original channels, weighted by their contrast.
  g=np.abs(np.gradient(profiles,axis=1))
  colour_gradient=np.gradient(colour,axis=1)
  cg=np.sqrt(.10*colour_gradient[:,:,0]**2 + colour_gradient[:,:,1]**2 + colour_gradient[:,:,2]**2)
  for which,lo,hi,expected in [('outer',pad-65,pad-13,pad-30),('inner',pad+12,pad+43,pad+24)]:
   gg=np.linalg.norm(colour_gradient,axis=2) if which=='inner' else cg
   cand=[]
   for j,t in enumerate(along):
    peaks=np.where((gg[j,lo+1:hi-1]>=gg[j,lo:hi-2])&(gg[j,lo+1:hi-1]>=gg[j,lo+2:hi])&(gg[j,lo+1:hi-1]>max(1.,.05*gg[j,lo:hi].max())))[0]+lo+1
    for p in peaks:
     a,b,c=gg[j,p-1:p+2];delta=.5*(a-c)/(a-2*b+c) if abs(a-2*b+c)>1e-6 else 0
     cand.append((t,p+np.clip(delta,-.5,.5),gg[j,p]**.25))
   m,s=_frame_ransac(np.array(cand),expected,(5 if which=="inner" else 20),tol_px=1.5,iters=1500, max_slope=10.0/length)
   (outer if which=='outer' else inner)[side]=m
   dbg[side+'_'+which]=s
 srcouter=cv2.perspectiveTransform(quad_from_models(outer,*gray.shape[::-1])[None],np.linalg.inv(H))[0]
 srcinner=cv2.perspectiveTransform(quad_from_models(inner,*gray.shape[::-1])[None],np.linalg.inv(H))[0]
 return srcouter,srcinner,dbg,stripe


def _frame_ransac(candidates: np.ndarray, expected_inset: float,
                    inset_tol_px: float, tol_px: float = 2.5,
                    iters: int = 700, max_slope: float = .007) -> tuple[np.ndarray, float]:
    """Fit inset = slope * along + intercept, rewarding unique scan rows."""
    if len(candidates) < 20:
        raise ValueError("Too few supported transitions for GX keyline fit")
    rng = np.random.default_rng(0)
    t, d, weight = candidates.T
    best_model = None
    best_score = -1.0
    best_mask = None
    _, row_ids = np.unique(t, return_inverse=True)
    n_rows = int(row_ids.max()) + 1
    for _ in range(iters):
        i, j = rng.choice(len(t), 2, replace=False)
        if abs(t[i] - t[j]) < 2:
            continue
        slope = (d[j] - d[i]) / (t[j] - t[i])
        if abs(slope) > max_slope:
            continue
        intercept = d[i] - slope * t[i]
        residual = np.abs(d - (slope * t + intercept))
        mask = residual <= tol_px
        if not np.any(mask):
            continue
        # Multiple peaks from one profile must not count as multiple votes.
        votes = np.zeros(n_rows)
        np.maximum.at(votes, row_ids[mask], weight[mask])
        score = float(votes.sum())
        # Artwork/text can make a perfectly straight, well-supported line.
        # A real full-art GX keyline must additionally sit at the physical
        # frame inset.  This prior is deliberately soft so print variation is
        # tolerated, but it breaks ties with a footer or attack-rule line.
        mid_inset = slope * float(np.median(t)) + intercept
        score *= float(np.exp(-((mid_inset - expected_inset) / inset_tol_px) ** 2))
        if score > best_score:
            best_model, best_score, best_mask = np.array([slope, intercept]), score, mask
    if best_model is None:
        raise ValueError("No coherent GX keyline line found")
    # Iteratively select the strongest inlier per profile, then weighted-fit.
    model = best_model
    for _ in range(3):
        residual = np.abs(d - (model[0] * t + model[1]))
        mask = residual <= tol_px
        picked = []
        for row in np.unique(t[mask]):
            choices = np.flatnonzero(mask & (t == row))
            picked.append(choices[np.argmax(weight[choices])])
        picked = np.asarray(picked, int)
        if len(picked) < 20:
            break
        model = np.polyfit(t[picked], d[picked], 1, w=np.sqrt(weight[picked]))
    support = float(len(picked) / len(np.unique(t))) if len(picked) else 0.0
    return np.asarray(model, float), support

