"""Development-set sensitivity checks. Not an independent accuracy benchmark."""
import argparse,base64,json
from pathlib import Path
import cv2
import numpy as np
from edge_classifier import classify,DEFAULTS
from pixel_whitening import inspect
from review_corners import read_document

def iou(a,b):
    x,y,w,h=a;u,v,p,q=b
    area=max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))
    return area/(w*h+p*q-area) if area else 0

def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('results');p.add_argument('--out',required=True)
    a=p.parse_args();records=json.loads(Path(a.results).read_text());cv2.setNumThreads(2)
    r=next(r for r in records if r['number']==5)
    sensitivity=[]
    for key in DEFAULTS:
        for factor in (.9,1.1):
            settings=dict(DEFAULTS);settings[key]*=factor
            predicted=classify(r,settings)
            sensitivity.append(dict(parameter=key,factor=factor,likely=[d['id'] for d in predicted['decisions'] if d['classification']=='likely_whitening']))
    repeatable=all(classify(r)==classify(r) for _ in range(3))
    batch,_=read_document(a.source);stress=[]
    for number in (4,5):
        c=next(c for c in batch['cards'] if c['number']==number)
        s=next(i['src'] for i in c['images'] if i['label']=='Corner close-ups')
        image=cv2.imdecode(np.frombuffer(base64.b64decode(s.split(',')[1]),np.uint8),1)
        base=next(r for r in records if r['number']==number)
        labeled={1,2,3} if number==4 else {17,19,34,63,12}
        anchors=[c for c in base['candidates']+base['excluded_candidates'] if c['id'] in labeled]
        for variant in ('original','darker10','brighter10','mild_blur'):
            pixels=image.copy()
            if variant=='darker10':pixels=(pixels.astype(float)*.9).astype(np.uint8)
            if variant=='brighter10':pixels=np.clip(pixels.astype(float)*1.1,0,255).astype(np.uint8)
            if variant=='mild_blur':pixels=cv2.GaussianBlur(pixels,(3,3),.6)
            result,_=inspect(pixels,.6,.02,'triangle',True,True,True,True)
            allc=result['candidates']+result['excluded_candidates']
            decisions={d['id']:d for d in classify(result)['decisions']}
            for anchor in anchors:
                matches=[(iou(anchor['box_in_corner'],c['box_in_corner']),c) for c in allc if c['corner']==anchor['corner']]
                score,match=max(matches,key=lambda t:t[0]) if matches else (0,None)
                outcome=decisions[match['id']]['classification'] if score>=.2 else 'not_detected'
                stress.append(dict(card=number,anchor_id=anchor['id'],variant=variant,iou=score,classification=outcome))
    output=dict(repeated_classification_identical=repeatable,threshold_sensitivity=sensitivity,
                image_sensitivity=stress,notes='Same labeled development photos; not new independent labels. Matching requires bounding-box IoU >=0.2.')
    Path(a.out).write_text(json.dumps(output,indent=2))
    print('Threshold variations retaining exactly four card5 positives:',sum(set(s['likely'])=={17,19,34,63} for s in sensitivity),'/',len(sensitivity))
    for row in stress:print(row['card'],row['anchor_id'],row['variant'],row['classification'])

if __name__=='__main__':main()
