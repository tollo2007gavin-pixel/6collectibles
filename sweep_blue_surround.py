"""Offline sensitivity sweep, not an accuracy evaluation."""
import argparse, base64, json
from pathlib import Path
import cv2
import numpy as np
from review_corners import read_document
from pixel_whitening import inspect

def main():
    p=argparse.ArgumentParser()
    p.add_argument('review'); p.add_argument('--out',required=True)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    batch,_=read_document(a.review)
    cv2.setNumThreads(2)
    measurements=[]; summary=[]
    for radius in [.01,.02,.03,.04]:
        candidates=[]
        for card in batch['cards']:
            source=next((im for im in card['images'] if im['label']=='Corner close-ups'),None)
            if source is None: continue
            raw=base64.b64decode(source['src'].split(',',1)[1],validate=True)
            sheet=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            result,_=inspect(sheet,1.1,radius) # Keep every candidate while measuring.
            for c in result['candidates']:
                candidates.append(dict(card=card['number'],**c))
        measurements.append(dict(radius_fraction=radius,candidates=candidates))
        for threshold in [.70,.75,.80,.85,.90]:
            def excluded(c):
                r=c['surrounding_blue']
                return (r['complete_ring'] and
                        r['blue_fraction']>=threshold and r['minimum_sector_blue_fraction']>=.35)
            removed=[c for c in candidates if excluded(c)]
            anchors=[dict(card=c['card'],id=c['id'],removed=excluded(c),
                          protected_by_size=False,
                          **c['surrounding_blue']) for c in candidates
                     if (c['card']==4 and c['id'] in [1,2,3]) or (c['card']==5 and c['id']==17)]
            summary.append(dict(radius_fraction=radius,threshold=threshold,total=len(candidates),
                                removed=len(removed),retained=len(candidates)-len(removed),anchors=anchors))
    (out/'measurements.json').write_text(json.dumps(measurements,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    for s in summary: print(s['radius_fraction'],s['threshold'],s['removed'],s['retained'])

if __name__=='__main__': main()
