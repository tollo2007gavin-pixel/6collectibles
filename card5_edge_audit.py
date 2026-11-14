"""One-card diagnostic review. User labels are comparison data, never detector inputs."""
import base64,hashlib,html,json
from pathlib import Path
import sys
import cv2
import numpy as np
from review_corners import read_document
from pixel_whitening import inspect

def main():
    batch,_=read_document(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
    card=next(c for c in batch['cards'] if c['number']==5)
    source=next(i for i in card['images'] if i['label']=='Corner close-ups')
    labels=json.loads(Path(sys.argv[3]).read_text())
    raw=base64.b64decode(source['src'].split(',',1)[1])
    if labels['card']!=5 or hashlib.sha256(raw).hexdigest()!=labels['source_sha256']:
        raise ValueError('Human labels belong to a different card image')
    positives=set(labels['positive_candidates']);negatives=set(labels['confirmed_sleeve_candidates'])
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
    result,_=inspect(image,.6,.02,'triangle',True,True,True)
    original_ids={12,14,15,16,17,19,20,22,23,24,29,34,44,54,56,58,61,62,63,68}
    assert {c['id'] for c in result['candidates']}==original_ids
    rows=[];overlay=image.copy()
    for c in result['candidates']:
        # Labels below were provided by the user for this exact saved card.
        label='Confirmed whitening' if c['id'] in positives else 'Confirmed sleeve' if c['id'] in negatives else 'Glare or inconclusive (not individually labeled)'
        association=c['edge']['association']
        color=(0,140,255) if association=='off_card_uncertain' else (0,0,255)
        x,y,w,h=c['box_in_sheet'];cv2.rectangle(overlay,(x,y),(x+w,y+h),color,1)
        cv2.putText(overlay,str(c['id']),(x,max(12,y-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
        rows.append(f'<tr><td>{c["id"]}</td><td>{label}</td><td>{association.replace("_"," ")}</td>'
                    f'<td>{c["edge"]["possible_card_overlap"]:.0%}</td><td>{c["edge"]["min_outside_distance"]:.1f}</td></tr>')
        c['human_label']=label
    cv2.imwrite(str(out/'card5-location.png'),overlay)
    encoded=base64.b64encode((out/'card5-location.png').read_bytes()).decode()
    page='''<!doctype html><meta charset="utf-8"><title>Card 5 edge evidence</title>
<style>body{font:17px system-ui;max-width:1100px;margin:25px auto;padding:15px}img{max-width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}@media(max-width:750px){.pair{display:block}}</style>
<h1>Card 5: edge evidence audit</h1><p>Your labels: 17, 19, 34, 63 are whitening; 12 is sleeve contamination. Other spots remain glare or inconclusive, not individually labeled negatives.</p>
<p>Orange: outside all raw card estimates but inside the safety margin. Red: touches at least one possible card interior. Neither color proves whitening. This is diagnostic only: candidate retention and safety margins are unchanged.</p>'''
    page+=f'<div class="pair"><img src="{html.escape(source["src"],quote=True)}" alt="Original card corners"><img src="data:image/png;base64,{encoded}" alt="Location evidence"></div>'
    page+='<table><tr><th>ID</th><th>Your judgment</th><th>Measured location</th><th>Card overlap</th><th>Minimum distance outside (px)</th></tr>'+''.join(rows)+'</table>'
    page+='<p>Only 4 positive examples and 1 confirmed negative are available. Measurements are fitted to this photo; no accuracy percentage or generalization claim is justified.</p>'
    (out/'review.html').write_text(page,encoding='utf-8')
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert next(c for c in result['candidates'] if c['id']==12)['edge']['association']=='off_card_uncertain'
    assert all(c['edge']['association']=='card_associated' for c in result['candidates'] if c['id'] in {17,19,34,63})
    print('Card 5: same 20 candidates; 12 is off-card uncertain; all 4 positives card-associated.')

if __name__=='__main__':main()
