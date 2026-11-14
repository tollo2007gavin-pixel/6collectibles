"""Offline comparison on existing sheets. Counts are not ground-truth accuracy."""
import argparse,base64,json,time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import edge_uncertainty
from pixel_whitening import inspect,corner_tiles
from review_corners import read_document

def main():
    p=argparse.ArgumentParser();p.add_argument('review');p.add_argument('--out',required=True)
    a=p.parse_args();output=Path(a.out);output.mkdir(parents=True,exist_ok=False)
    batch,_=read_document(a.review);cv2.setNumThreads(2)
    originals=[]
    for c in batch['cards']:
        source=next((i for i in c['images'] if i['label']=='Corner close-ups'),None)
        if source:
            raw=base64.b64decode(source['src'].split(',',1)[1])
            originals.append((c['number'],cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)))
    reports=[];estimate=edge_uncertainty.estimate
    for margin in (.024,.04,.06):
        started=time.perf_counter();rows=[]
        for number,image in originals:
            with patch.object(edge_uncertainty,'estimate',side_effect=lambda c,n,**kw:estimate(c,n,margin,**kw)):
                result,_=inspect(image,.6,.02,'triangle',True,True)
            rows.append(dict(number=number,retained=[c['id'] for c in result['candidates']],
                             background=[c['id'] for c in result['excluded_candidates'] if c['reason_code']=='B'],
                             corners=result['corners']))
        reports.append(dict(margin_fraction=margin,seconds=time.perf_counter()-started,cards=rows))
        print(margin,sum(len(r['retained']) for r in rows),next(len(r['retained']) for r in rows if r['number']==5),flush=True)
    (output/'sensitivity.json').write_text(json.dumps(reports,indent=2))
    # Sensitivity, not new independent labels: known sheets under mild exposure,
    # blur and downsampling changes. Evaluate the edge stage alone at fixed spots.
    stress=[]
    anchors={4:[('TR',1,[249,184,21,24]),('TR',2,[236,205,13,14]),('TR',3,[217,224,12,13])],
             5:[('TR',17,[166,193,16,16])]}
    for number,image in originals:
        if number not in anchors:continue
        for variant in ['original','darker','brighter','blur','resample']:
            altered=image.copy()
            if variant=='darker':altered=(image.astype(float)*.7).astype(np.uint8)
            if variant=='brighter':altered=np.clip(image.astype(float)*1.15+10,0,255).astype(np.uint8)
            if variant=='blur':altered=cv2.GaussianBlur(image,(5,5),1.2)
            if variant=='resample':altered=cv2.resize(cv2.resize(image,None,fx=.5,fy=.5), (image.shape[1],image.shape[0]))
            crops={n:c for n,_,_,c in corner_tiles(altered)}
            for corner,identifier,box in anchors[number]:
                info,bg,zone=estimate(crops[corner],corner,.04)
                x,y,w,h=box
                stress.append(dict(card=number,id=identifier,variant=variant,edge_status=info['status'],
                    entire_box_outside=bool(bg[y:y+h,x:x+w].all())))
    (output/'anchor_stress.json').write_text(json.dumps(stress,indent=2))

if __name__=='__main__':main()
