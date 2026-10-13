"""Evaluate triage without counting abstentions as correct damage detections."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

def score(labels, records):
    matrix=Counter();unlabeled=0;by_card=defaultdict(list)
    for r in records:
        cid=r.get('card_id')
        if cid not in labels:unlabeled+=1;continue
        outcome=r.get('decision')
        if outcome not in ('clean','whitening_detected','needs_review'):raise ValueError('Invalid triage decision')
        label=labels[cid]
        if label not in ('clean','whitening'):raise ValueError('Invalid human label')
        matrix[(label,outcome)]+=1;by_card[cid].append(outcome)
    repeated={cid:v for cid,v in by_card.items() if len(v)>1}
    return {'evaluations':sum(matrix.values()),'cards_evaluated':len(by_card),
        'missing_labeled_cards':sorted(set(labels)-set(by_card)),'unlabeled_records':unlabeled,
        'confusion_counts':{label:{o:matrix[(label,o)] for o in ('clean','whitening_detected','needs_review')} for label in ('clean','whitening')},
        'repeated_cards':len(repeated),'fully_consistent_cards':sum(len(set(v))==1 for v in repeated.values()),
        'uncertain_is_not_counted_as_detected':True}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--labels',default='labels.csv')
    ap.add_argument('--results',default='cards/whitening_v2');ap.add_argument('--out')
    a=ap.parse_args();labels={r['card_id']:r['label'] for r in csv.DictReader(Path(a.labels).open(encoding='utf-8-sig'))}
    records=[]
    for f in Path(a.results).rglob('*.json'):
        r=json.loads(f.read_text(encoding='utf-8'))
        if isinstance(r,dict) and 'decision' in r and 'card_id' in r:records.append(r)
    report=score(labels,records);text=json.dumps(report,indent=2)
    if a.out:Path(a.out).write_text(text,encoding='utf-8')
    print(text)

if __name__=='__main__':main()
