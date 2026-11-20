"""Offline classifier calibration from human region labels; never changes production.

The cache fixes candidate extraction and patch measurements. This tool tunes only
core classifier thresholds. Validation is a separate, frozen-configuration action.
"""
import argparse
import gzip
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import urlparse

from edge_classifier import DEFAULTS, PATCH_DEFAULTS, VERSION, classify
from pixel_active_settings import active_thresholds

SCHEMA='pixel-whitening-labels-v1'
CONFIG_SCHEMA='pixel-whitening-frozen-v1'
EXPOSURE_SCHEMA='pixel-whitening-validation-exposure-v1'
CORNERS={'TL','TR','BL','BR'}
MATCHING=dict(iou=.1,smaller_overlap=.5,max_area_ratio=25)
CODE_FILES=('edge_classifier.py','pixel_whitening.py','edge_uncertainty.py','interior_art.py','patch_evidence.py')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def code_fingerprint():
    # Normalize newlines so Windows and GitHub Linux agree on the same source.
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest()
            for name in CODE_FILES+('pixel_training.py',)}


def check_cached_provenance(result):
    if result.get('analysis_mode')!='whitening_classification' or result.get('whitening_classification',{}).get('version')!=VERSION:
        raise ValueError('Labeling is allowed, but tuning requires a current edge_classifier run with patch measurements; replay the sheets first')
    recorded=result.get('code_sha256',{})
    for name in CODE_FILES:
        path=Path(__file__).with_name(name)
        lf=path.read_text(encoding='utf-8').replace('\r\n','\n')
        accepted={hashlib.sha256(data).hexdigest() for data in (path.read_bytes(),lf.encode(),lf.replace('\n','\r\n').encode())}
        if recorded.get(name) not in accepted:
            raise ValueError(f'Cached features come from different code ({name}); replay the photos using the current edge_classifier before tuning')


def listing_key(value):
    value=str(value).strip().lower()
    if not value:return ''
    match=re.fullmatch(r'v1\|(\d+)\|[^|]+',value)
    if match:return match[1]
    if value.isdigit():return value
    if value.startswith(('https://','http://')):
        parts=urlparse(value).path.rstrip('/').split('/')
        if 'itm' in parts and parts[-1].isdigit():return parts[-1]
    return value


def aliases(card):
    return sorted({'group:'+card['group_id'].strip().casefold(),
                   'listing:'+listing_key(card['listing']),
                   'image:'+card['source_sha256'].lower()})


def box_valid(box,size):
    return (isinstance(box,list) and len(box)==4 and
            all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box) and
            min(box[:2])>=0 and min(box[2:])>0 and box[0]+box[2]<=size and box[1]+box[3]<=size)


def overlap(a,b):
    x,y,w,h=a;u,v,p,q=b
    inter=max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))
    if not inter:return 0.,0.,float('inf')
    aa=w*h;bb=p*q
    return inter/(aa+bb-inter),inter/min(aa,bb),max(aa,bb)/min(aa,bb)


def matches(a,b,policy=MATCHING):
    if a['corner']!=b['corner']:return False
    iou,small,ratio=overlap(a['box'],b['box'])
    return iou>=policy['iou'] or (small>=policy['smaller_overlap'] and ratio<=policy['max_area_ratio'])


def validate_cards(cards):
    """Validate data and reject split leakage through transitive identity aliases."""
    if not isinstance(cards,list) or not cards:raise ValueError('Dataset has no cards')
    normalized=[];seen={};alias_owners={};parent=list(range(len(cards)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(a)]=find(b)
    for i,source in enumerate(cards):
        c=copy.deepcopy(source)
        if not isinstance(c,dict):raise ValueError('Each card must be an object')
        if c.get('split') not in ('development','validation'):raise ValueError('Choose development or validation for each card')
        for name in ('group_id','listing'):
            if not isinstance(c.get(name),str) or not c[name].strip():raise ValueError('Missing physical-card group or listing identity')
        if not re.fullmatch('[0-9a-fA-F]{64}',str(c.get('source_sha256',''))):raise ValueError('Missing valid source image SHA256')
        r=c.get('result',{})
        if not isinstance(r,dict):raise ValueError('Cached result must be an object')
        dims=r.get('measurement_image_size',[])
        check_cached_provenance(r)
        if len(dims)!=2 or any(type(v) is not int for v in dims):raise ValueError('Missing saved measurement dimensions')
        size=(dims[0]-10)/2
        if size<32 or size!=int(size) or dims[1]!=size*2+50:raise ValueError('Invalid saved corner sheet layout')
        if r.get('source_sha256',c['source_sha256']).lower()!=c['source_sha256'].lower():raise ValueError('Result image does not match annotation image')
        if listing_key(r.get('listing',c['listing']))!=listing_key(c['listing']):raise ValueError('Result listing does not match annotations')
        if not isinstance(r.get('candidates'),list) or not isinstance(r.get('excluded_candidates'),list):raise ValueError('Missing cached candidates')
        ids=set()
        for candidate in r['candidates']+r['excluded_candidates']:
            if not isinstance(candidate,dict):raise ValueError('Cached candidate must be an object')
            if type(candidate.get('id')) is not int or candidate['id'] in ids:raise ValueError('Candidate IDs must be unique integers per sheet')
            ids.add(candidate['id'])
            if candidate.get('corner') not in CORNERS or not box_valid(candidate.get('box_in_corner'),size):raise ValueError('Invalid candidate coordinates')
            for patch in candidate.get('shape',{}).get('patch_evidence',{}).get('scales',[]):
                if 'box_in_corner' in patch and not box_valid(patch['box_in_corner'],size):raise ValueError('Invalid supporting patch coordinates')
        try:classify(r)
        except (KeyError,TypeError,ValueError,ZeroDivisionError) as exc:raise ValueError('Incomplete classifier feature cache') from exc
        complete=c.get('complete_corners',[])
        if not isinstance(complete,list) or any(x not in CORNERS for x in complete):raise ValueError('Invalid completed corners')
        c['complete_corners']=sorted(set(complete))
        marks=c.get('annotations',[])
        if not isinstance(marks,list):raise ValueError('Annotations must be a list')
        mark_ids=set()
        for mark in marks:
            if not isinstance(mark,dict):raise ValueError('Annotation must be an object')
            if mark.get('label') not in ('whitening','not_whitening','uncertain'):raise ValueError('Unknown annotation label')
            if mark.get('corner') not in CORNERS or not box_valid(mark.get('box'),size):raise ValueError('Annotation outside its corner')
            if not isinstance(mark.get('id'),(str,int)) or str(mark['id']) in mark_ids:raise ValueError('Annotation IDs must be unique')
            mark_ids.add(str(mark['id']))
        # Explicit export policy: untouched candidates on a reviewed card are
        # negative. Drawn positives and uncertain regions take precedence.
        if c.get('positive_only_review'):
            if not marks and not str(c.get('notes','')).strip():
                raise ValueError('Positive-only review requires marks or a note; untouched cards remain unreviewed')
            for region in reviewed_candidate_boxes(r):
                if any(matches(region,a) or (a.get('candidate_id')==region['id'] and a['corner']==region['corner']) for a in marks):continue
                identifier='automatic-negative-'+str(region['id'])
                if identifier in mark_ids:raise ValueError('Automatic annotation ID conflicts with a manual mark')
                marks.append(dict(id=identifier,corner=region['corner'],box=region['box'],label='not_whitening',candidate_id=region['id'],origin='untouched_candidate'))
                mark_ids.add(identifier)
        # Contradictory overlapping marks cannot silently hide false positives.
        for j,a in enumerate(marks):
            for b in marks[j+1:]:
                if matches(a,b) and a['label']!=b['label']:raise ValueError('Conflicting labels overlap; adjust or remove one mark')
                if matches(a,b) and a['label']==b['label']=='whitening':raise ValueError('Overlapping whitening marks may count one chip twice')
        c['annotations']=marks
        digest(c)  # rejects NaN/Infinity anywhere in supplied features
        for alias in aliases(c):
            if alias in alias_owners:union(i,alias_owners[alias])
            else:alias_owners[alias]=i
        key=c['source_sha256'].lower()
        if key in seen:
            previous=seen[key]
            # A copied file may have different display numbering; everything
            # affecting review, grouping or predictions must still agree.
            def payload(x):
                value={k:x.get(k) for k in ('listing','group_id','split','annotations','complete_corners','result')}
                value['result']={k:v for k,v in value['result'].items() if k not in ('number','human_review')}
                return value
            if digest(payload(previous))!=digest(payload(c)):raise ValueError('Conflicting duplicate image reviews')
        else:seen[key]=c;normalized.append(c)
    split_by_root={}
    for i,c in enumerate(cards):
        split_by_root.setdefault(find(i),set()).add(c['split'])
    if any(len(s)>1 for s in split_by_root.values()):raise ValueError('Same card/listing/image appears in development and validation')
    return normalized


def load_dataset(path, exposures=()):
    path=Path(path);files=sorted([*path.glob('*.json'),*path.glob('*.json.gz')]) if path.is_dir() else [path]
    cards=[]
    for file in files:
        data=json.loads((gzip.decompress(file.read_bytes()).decode('utf-8-sig') if file.name.endswith('.json.gz') else file.read_text(encoding='utf-8-sig')),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Non-finite JSON number')))
        if not isinstance(data,dict):raise ValueError(f'Expected a labeled dataset object: {file.name}')
        if data.get('schema') in (CONFIG_SCHEMA,EXPOSURE_SCHEMA):continue
        if data.get('schema')!=SCHEMA:raise ValueError(f'Not a labeled pixel dataset: {file.name}')
        cards.extend(data.get('cards',[]))
    if exposures:
        cards = retire_exposed_groups(cards, exposures)
    return validate_cards(cards)


def load_exposures(path):
    path=Path(path)
    if not path.is_dir():path=path.parent
    exposures=[]
    for file in sorted(path.glob('*.json')):
        data=json.loads(file.read_text(encoding='utf-8-sig'))
        if not isinstance(data,dict):continue
        if data.get('schema')==EXPOSURE_SCHEMA:
            if not isinstance(data.get('exposed_aliases'),list) or any(not isinstance(x,str) for x in data['exposed_aliases']):raise ValueError('Invalid validation exposure manifest')
            exposures.append(data)
    return exposures


def candidate_box(candidate, decision):
    patches = candidate.get('shape', {}).get('patch_evidence', {}).get('scales', [])
    if decision.get('reason') == 'compact_light_patch_supported_at_two_scales' and patches:
        return max(patches, key=lambda patch: patch['scale'])['box_in_corner']
    return candidate['box_in_corner']


def reviewed_candidate_boxes(result):
    """Use the exact boxes shown in the editor, even after thresholds change."""
    decisions = {d['id']:d for d in result['whitening_classification'].get('decisions', [])}
    for candidate in result['candidates'] + result['excluded_candidates']:
        identifier = candidate['id']
        yield dict(id=identifier, corner=candidate['corner'],
                   box=candidate_box(candidate, decisions.get(identifier, {})))


def prediction_regions(result,limits):
    source={c['id']:c for c in result['candidates']+result['excluded_candidates']}
    return [dict(id=d['id'],corner=d['corner'],box=candidate_box(source[d['id']],d),
                 classification=d['classification']) for d in classify(result,limits)['decisions']]


def score_cards(cards,limits,policy=MATCHING):
    totals=dict(true_positive_regions=0,false_negative_regions=0,false_positive_marks=0,
        positive_regions=0,explicit_negative_regions=0,completed_corners=0,
        ignored_unreviewed_marks=0,ignored_uncertain_marks=0,duplicate_chip_marks=0,
        extraction_missed_regions=0,inconclusive_positive_regions=0,artifact_positive_regions=0)
    details=[]
    for card in cards:
        labels=card['annotations'];positive=[a for a in labels if a['label']=='whitening']
        negative=[a for a in labels if a['label']=='not_whitening'];uncertain=[a for a in labels if a['label']=='uncertain']
        regions=prediction_regions(card['result'],limits)
        by_id={r['id']:r for r in regions}
        extracted=[]
        for candidate in card['result']['candidates']+card['result']['excluded_candidates']:
            base=dict(by_id[candidate['id']],box=candidate['box_in_corner'])
            extracted.append(base)
            for patch in candidate.get('shape',{}).get('patch_evidence',{}).get('scales',[]):
                if patch.get('status')=='measured' and 'box_in_corner' in patch:
                    extracted.append(dict(base,box=patch['box_in_corner']))
        predicted=[r for r in regions if r['classification']=='likely_whitening']
        counts={k:0 for k in totals}
        counts.update(positive_regions=len(positive),explicit_negative_regions=len(negative),completed_corners=len(card['complete_corners']))
        usable=[]
        for r in predicted:
            if any(matches(r,a,policy) for a in uncertain):counts['ignored_uncertain_marks']+=1
            else:usable.append(r)
        adjacency=[[i for i,a in enumerate(positive) if matches(r,a,policy)] for r in usable]
        assigned={}
        def augment(pred,visited):
            for label in adjacency[pred]:
                if label in visited:continue
                visited.add(label)
                if label not in assigned or augment(assigned[label],visited):assigned[label]=pred;return True
            return False
        for i in range(len(usable)):augment(i,set())
        counts['true_positive_regions']=len(assigned)
        counts['false_negative_regions']=len(positive)-len(assigned)
        assigned_preds=set(assigned.values())
        for i,r in enumerate(usable):
            if i in assigned_preds:continue
            if adjacency[i]:counts['duplicate_chip_marks']+=1
            elif any(matches(r,a,policy) for a in negative) or r['corner'] in card['complete_corners']:counts['false_positive_marks']+=1
            else:counts['ignored_unreviewed_marks']+=1
        for i,a in enumerate(positive):
            if i in assigned:continue
            matching=[r for r in extracted if matches(r,a,policy)]
            # Extraction misses are distinct from candidates whose classifier abstained.
            if not matching:counts['extraction_missed_regions']+=1
            elif any(r['classification']=='inconclusive' for r in matching):counts['inconclusive_positive_regions']+=1
            elif any(r['classification']=='likely_artifact' for r in matching):counts['artifact_positive_regions']+=1
        for k,v in counts.items():totals[k]+=v
        details.append(dict(number=card.get('number'),listing=card['listing'],source_sha256=card['source_sha256'],counts=counts))
    tp=totals['true_positive_regions'];fp=totals['false_positive_marks'];fn=totals['false_negative_regions']
    negative_coverage=bool(totals['explicit_negative_regions'] or totals['completed_corners'])
    precision=tp/(tp+fp) if tp+fp and negative_coverage else None
    recall=tp/(tp+fn) if tp+fn else None
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn and negative_coverage else None
    return dict(counts=totals,precision=precision,recall=recall,f1=f1,cards=details,
        scope='Reviewed regions only. Unmarked areas are negative only inside explicitly completed corners; uncertain marks are ignored. Precision and F1 require negative-region coverage.')


def training_fingerprint(cards):
    # Notes and validation records are deliberately absent from optimization inputs.
    rows=[dict(aliases=aliases(c),annotations=c['annotations'],complete_corners=c['complete_corners'],features=digest(c['result'])) for c in cards]
    return digest(sorted(rows,key=lambda r:r['aliases']))


def frozen_config(cards,limits,selection_status):
    result=dict(schema=CONFIG_SCHEMA,classifier_version=VERSION,code_fingerprint=code_fingerprint(),
        core_thresholds=dict(limits),fixed_patch_thresholds=dict(PATCH_DEFAULTS),matching_policy=dict(MATCHING),
        development_fingerprint=training_fingerprint(cards),training_aliases=sorted({a for c in cards for a in aliases(c)}),
        cached_provenance=sorted({digest(c['result'].get('code_sha256',{})) for c in cards}),
        selection_status=selection_status,
        limitation='Cached-feature calibration only: does not change candidate extraction, edge/artwork estimates or patch measurements.')
    result['config_sha256']=digest(result)
    return result


def proposals(current):
    yield 'baseline',dict(current)
    for name in current:
        for factor in (.8,1.2):
            values=dict(current);values[name]*=factor
            yield f'{name} x {factor:g}',values


def card_groups(cards):
    """Join all photos sharing a listing, image hash or physical-card group."""
    groups = []
    for card in cards:
        keys = set(aliases(card))
        members = [card]
        separate = []
        for existing_keys, existing_members in groups:
            if keys.intersection(existing_keys):
                keys.update(existing_keys)
                members.extend(existing_members)
            else:
                separate.append((existing_keys, existing_members))
        groups = separate + [(keys, members)]
    return groups


def enough_review(counts, groups):
    return (groups >= 3 and counts['positive_regions'] >= 3 and
            (counts['explicit_negative_regions'] >= 3 or counts['completed_corners'] >= 1))


def has_scored_labels(card):
    return bool(card['complete_corners'] or
                any(a['label'] in ('whitening','not_whitening') for a in card['annotations']))


def calibrate(cards,tune=True,current=None):
    development=[c for c in cards if c['split']=='development']
    if not development:raise ValueError('No development cards; validation answers cannot be used to tune')
    if current is None:current=active_thresholds()
    baseline=score_cards(development,current);counts=baseline['counts']
    groups=sum(any(has_scored_labels(c) for c in members) for _,members in card_groups(development))
    enough=enough_review(counts, groups)
    chosen=dict(current);best=baseline;trials=[];status='baseline_only'
    if tune and enough:
        for name,limits in proposals(current):
            score=baseline if name=='baseline' else score_cards(development,limits)
            trials.append(dict(setting=name,thresholds=limits,precision=score['precision'],recall=score['recall'],f1=score['f1']))
            if score['recall']>=baseline['recall'] and score['f1']>best['f1']+1e-12:
                chosen=limits;best=score
        status='development_improvement' if chosen!=current else 'no_development_improvement'
    elif tune:status='insufficient_labels'
    return dict(mode='tune' if tune else 'baseline',selection_status=status,development_cards=len(development),
        baseline=baseline,selected=best,trials=trials,
        warning='Validation labels were not scored. Development gains are not independent accuracy. No production files were changed.',
        minimum_for_tuning='At least 3 physical-card groups, 3 whitening regions, and 3 explicit negative regions or 1 fully reviewed corner.'),frozen_config(development,chosen,status)


def validate_frozen(cards,config,exposures=()):
    if config.get('schema')!=CONFIG_SCHEMA:raise ValueError('Validation requires a frozen configuration from baseline/tune')
    payload={k:v for k,v in config.items() if k!='config_sha256'}
    if config.get('config_sha256')!=digest(payload):raise ValueError('Frozen configuration was modified')
    if config.get('code_fingerprint')!=code_fingerprint():raise ValueError('Code changed since configuration was frozen; freeze and validate a compatible version')
    if config.get('classifier_version')!=VERSION or config.get('fixed_patch_thresholds')!=PATCH_DEFAULTS or config.get('matching_policy')!=MATCHING:raise ValueError('Frozen classifier or matching policy is incompatible')
    limits=config.get('core_thresholds',{})
    if set(limits)!=set(DEFAULTS) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in limits.values()):raise ValueError('Invalid frozen thresholds')
    validation=[c for c in cards if c['split']=='validation']
    if not validation:raise ValueError('No reserved validation cards')
    seen=set(config['training_aliases'])
    if any(seen.intersection(aliases(c)) for c in validation):raise ValueError('Reserved cards overlap the frozen training identities')
    exposed={a for record in exposures for a in record['exposed_aliases']}
    if any(exposed.intersection(aliases(c)) for c in validation):raise ValueError('These validation cards were already exposed; move them to development and reserve fresh cards')
    report=score_cards(validation,limits,config['matching_policy'])
    report.update(mode='validate',config_sha256=config['config_sha256'],validation_cards=len(validation),
        warning='Validation answers are now exposed. If these failures guide another change, move these cards to development and reserve fresh cards. The tool cannot detect undisclosed duplicate cards or external reuse of answers.')
    exposure=dict(schema=EXPOSURE_SCHEMA,config_sha256=config['config_sha256'],exposed_aliases=sorted({a for c in validation for a in aliases(c)}),
        annotation_fingerprint=training_fingerprint(validation),status='exposed_do_not_reuse_as_fresh_validation')
    return report,exposure


def retire_exposed_groups(cards, exposures):
    exposed = {key for record in exposures for key in record['exposed_aliases']}
    if not exposed:
        return cards
    retired = []
    for keys, members in card_groups(cards):
        for card in members:
            if card['split'] == 'validation' and exposed.intersection(keys):
                retired.append(dict(card, split='development'))
            else:
                retired.append(card)
    return retired


def improve(cards, exposures=()):
    """Choose settings on pooled development groups, then score one frozen choice."""
    cards = retire_exposed_groups(cards, exposures)
    current = active_thresholds()
    development = [c for c in cards if c['split'] == 'development']
    reserved_groups = [members for _, members in card_groups([c for c in cards if c['split'] == 'validation'])
                       if any(has_scored_labels(c) for c in members)]
    reserved = [c for members in reserved_groups for c in members]
    report = dict(mode='improve', active_thresholds=current, decision='need_reviews',
                  development_cards=len(development), validation_cards=0,
                  fresh_reviewed_reserved_cards=len(reserved))
    files = {}
    if not development:
        report['message'] = 'Add reviewed development cards to compare settings.'
        return report, files
    tuning, frozen = calibrate(development, current=current)
    report['development'] = tuning
    files['frozen-config.json'] = frozen
    if tuning['selection_status'] == 'insufficient_labels':
        report['message'] = tuning['minimum_for_tuning']
        return report, files
    if tuning['selection_status'] != 'development_improvement':
        report.update(decision='keep_current', message='Keep the current settings. No trial improved the development score while preserving chip recall.')
        return report, files
    report['threshold_changes'] = {
        name: dict(live=current[name], proposed=value)
        for name, value in frozen['core_thresholds'].items() if value != current[name]
    }
    coverage = dict(positive_regions=sum(a['label']=='whitening' for c in reserved for a in c['annotations']),
                    explicit_negative_regions=sum(a['label']=='not_whitening' for c in reserved for a in c['annotations']),
                    completed_corners=sum(len(c['complete_corners']) for c in reserved))
    if not enough_review(coverage, len(reserved_groups)):
        report.update(decision='need_fresh_cards', message='Review another fresh batch. Reserved cards are saved for a comparison with at least three physical-card groups, three marked chips, and three negative regions or one fully reviewed corner.')
        return report, files
    candidate, exposure = validate_frozen(reserved, frozen, exposures)
    live = score_cards(reserved, current)
    counts = candidate['counts']
    gained = counts['true_positive_regions'] - live['counts']['true_positive_regions']
    added = counts['false_positive_marks'] - live['counts']['false_positive_marks']
    report.update(validation_cards=len(reserved), live=live, proposed=candidate,
                  config_sha256=frozen['config_sha256'], chips_found_change=gained, false_alarms_change=added)
    files['validation-exposure.json'] = exposure
    if gained >= 0 and added <= 0 and (gained > 0 or added < 0):
        report.update(decision='recommend_update', message='The proposed settings improved the reserved-card comparison. They caught more chips or reduced false alarms without worsening the other count.')
        files['recommended-settings.json'] = frozen['core_thresholds']
    else:
        report.update(decision='keep_current', message='Keep the current settings. The proposed settings tied, lost chips, or added false alarms on reserved cards.')
    return report, files


def improvement_markdown(report):
    lines = ['# Pixel whitening review result', '', report['message'], '',
             f"Used {report['development_cards']} development cards and scored {report['validation_cards']} fresh reserved cards."]
    tuning = report.get('development')
    if tuning and tuning['selection_status'] == 'development_improvement':
        before, after = (tuning[name]['counts'] for name in ('baseline','selected'))
        lines.extend(['', f"On development cards, live settings found {before['true_positive_regions']} chips with "
                      f"{before['false_positive_marks']} false alarms. Proposed settings found {after['true_positive_regions']} "
                      f"chips with {after['false_positive_marks']} false alarms. These cards were used to choose the settings."])
        for name, change in report['threshold_changes'].items():
            lines.append(f"\nProposed change for `{name}` from {change['live']:g} to {change['proposed']:g}.")
    if 'live' in report:
        lines.extend(['', '| Reserved-card result | Live | Proposed |',
                      '|---|---:|---:|'])
        for name, key in [('Chips found', 'true_positive_regions'), ('Chips missed', 'false_negative_regions'),
                          ('False alarms', 'false_positive_marks'), ('Misses without a matching candidate', 'extraction_missed_regions')]:
            lines.append(f"| {name} | {report['live']['counts'][key]} | {report['proposed']['counts'][key]} |")
        lines.extend(['', 'Counts refer to marked chips and alarms in reviewed areas. Each settings choice was scored on the same reserved cards. A small sample can change substantially with a single chip.'])
    return '\n'.join(lines) + '\n'


def markdown(report):
    scores=report.get('selected',report);c=scores['counts']
    fmt=lambda v:'not measurable' if v is None else f'{v:.1%}'
    return ('# Pixel whitening '+report['mode']+' report\n\n'+report.get('warning','')+'\n\n'
        +f"Detected reviewed whitening regions: {c['true_positive_regions']} / {c['positive_regions']}.\n\n"
        +f"Missed regions: {c['false_negative_regions']}; false-positive marks in reviewed areas: {c['false_positive_marks']}.\n\n"
        +f"Precision: {fmt(scores['precision'])}; recall: {fmt(scores['recall'])}; F1: {fmt(scores['f1'])}. These are not whole-batch accuracy.\n\n"
        +f"Unreviewed predictions ignored: {c['ignored_unreviewed_marks']}; uncertain predictions ignored: {c['ignored_uncertain_marks']}.\n\n"
        +f"Misses with no matching extracted candidate: {c['extraction_missed_regions']}.\n\n"
        +scores['scope']+'\n\n'+report.get('minimum_for_tuning','')+'\n\n'
        +'Settings affect cached core classification only. Candidate extraction and patch evidence are fixed. Results do not automatically alter the active pipeline.\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('baseline','tune','validate','improve'))
    parser.add_argument('--dataset',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--config')
    args=parser.parse_args(argv)
    try:
        exposures=load_exposures(args.dataset) if args.action in ('validate','improve') else []
        cards=load_dataset(args.dataset, exposures if args.action=='improve' else ())
        if args.action=='improve':
            report,files=improve(cards,exposures)
            summary=improvement_markdown(report)
        elif args.action=='validate':
            if not args.config:raise ValueError('Provide --config for frozen validation')
            config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
            report,extra=validate_frozen(cards,config,exposures)
            files={'validation-exposure.json':extra}
            summary=markdown(report)
        else:
            report,extra=calibrate(cards,args.action=='tune')
            files={'frozen-config.json':extra}
            summary=markdown(report)
        output=Path(args.out);output.mkdir(parents=True,exist_ok=False)
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        for name,value in files.items():
            (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
        (output/'REPORT.md').write_text(summary,encoding='utf-8')
        print(summary)
    except (ValueError,KeyError,TypeError,OSError) as exc:parser.error(str(exc))


if __name__=='__main__':main()
