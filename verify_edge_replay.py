"""Check deterministic saved-image results and exact legacy decision fallback."""
import argparse,base64,json,time
from pathlib import Path
import cv2
import numpy as np
from pixel_whitening import inspect
from review_corners import read_document

def decisions(r):
    return ([c['id'] for c in r['candidates']],[(c['id'],c['reason_code']) for c in r['excluded_candidates']])

def main():
    p=argparse.ArgumentParser();p.add_argument('review');p.add_argument('baseline');p.add_argument('enhanced');p.add_argument('--out',required=True)
    a=p.parse_args();cv2.setNumThreads(2)
    batch,_=read_document(a.review)
    baseline={r['number']:r for r in json.loads(Path(a.baseline).read_text())}
    enhanced={r['number']:r for r in json.loads(Path(a.enhanced).read_text())}
    rows=[];t=time.perf_counter()
    for card in batch['cards']:
        source=next((i for i in card['images'] if i['label']=='Corner close-ups'),None)
        if not source:continue
        image=cv2.imdecode(np.frombuffer(base64.b64decode(source['src'].split(',',1)[1]),np.uint8),cv2.IMREAD_COLOR)
        old,_=inspect(image,.6,.02,'triangle',True,False)
        new,_=inspect(image,.6,.02,'triangle',True,True)
        n=card['number']
        assert decisions(old)==decisions(baseline[n]),f'Legacy changed: {n}'
        assert decisions(new)==decisions(enhanced[n]),f'Non-repeatable: {n}'
        before={c['id'] for c in old['candidates']};after={c['id'] for c in new['candidates']}
        assert after<=before
        removed=before-after
        assert removed <= {c['id'] for c in new['excluded_candidates'] if c['reason_code']=='B'}
        if n==5:assert 17 in after
        if n==4:assert {1,2}<=after and 3 not in before and 3 not in after
        rows.append(dict(card=n,before=len(before),after=len(after),newly_excluded=sorted(removed)))
    Path(a.out).write_text(json.dumps(dict(exact_legacy_fallback=True,repeatable_decisions=True,
        seconds=time.perf_counter()-t,cards=rows),indent=2))
    print('Exact fallback and repeatability verified on',len(rows),'sheets')

if __name__=='__main__':main()
