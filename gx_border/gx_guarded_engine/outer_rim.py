"""Disambiguate a physical foil rim from its inner printed dark band.

Only runs on a proposed holder relocation. Four observed transitions must
agree: entry into a bright rim, dark band, bright shoulder, then GX keyline.
No card names, ideal margins or extrapolated corner measurements are used.
"""
import numpy as np
from .holder_topology import FRAME,sample
from .tracks import project

def profile(lab,inv,k,c):
    a,b=FRAME[[k,(k+1)%4]];v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
    t=np.linspace(.12,.88,161);d=np.polynomial.polynomial.polyval(t-.5,c['track']['coefficients'])
    observed=np.zeros(len(t),bool)
    for seg in c['track'].get('segments',[]):
        p=project(seg,np.linalg.inv(inv));ts=(p-a)@v/(v@v)
        observed|=(t>=ts.min())&(t<=ts.max())
    colors,valid=sample(lab,a+t[:,None,None]*v+(d[:,None,None]+np.array([-3.,3.])[None,:,None])*n,inv)
    return dict(t=t,d=d,delta=colors[:,1,0]-colors[:,0,0],valid=observed&valid.all(1))

def choose_rim(lab,inv,k,selected,candidates):
    p=profile(lab,inv,k,selected)
    if p['valid'].sum()<54 or np.mean(p['delta'][p['valid']]<-4)<.8:
        return selected,None,None
    rows=[(c,profile(lab,inv,k,c)) for c in candidates]
    matches=[]
    for edge,e in rows:
        gap=selected['offset']-edge['offset']
        if not 9<=gap<=24 or edge['offset']<=8:continue
        for shoulder,s in rows:
            if not 9<=shoulder['offset']-selected['offset']<=24:continue
            for keyline,z in rows:
                if not 7<=keyline['offset']-shoulder['offset']<=23:continue
                common=p['valid']&e['valid']&s['valid']&z['valid']
                if common.sum()<54:continue
                gaps=[p['d']-e['d'],s['d']-p['d'],z['d']-s['d']]
                polarity=(e['delta']>4)&(p['delta']<-4)&(s['delta']>4)&(z['delta']<-4)
                sequence=polarity&(gaps[0]>=9)&(gaps[0]<=24)&(gaps[1]>=9)&(gaps[1]<=24)&(gaps[2]>=7)&(gaps[2]<=23)
                fraction=float(np.mean(sequence[common]))
                if fraction>=.8:
                    matches.append((edge,keyline,dict(common_profiles=int(common.sum()),sequence_fraction=fraction,
                        old_offset=selected['offset'],edge_offset=edge['offset'],keyline_offset=keyline['offset'],
                        reason='four observed GX transitions place the physical rim before the selected printed dark band')))
    if len(matches)!=1:return selected,None,dict(ambiguous_sequences=len(matches)) if matches else None
    return matches[0]
