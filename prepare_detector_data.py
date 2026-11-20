"""Prepare native-resolution detector corners from local portable box reviews.

Loads the original embedded photo by its SHA-256. Labels use the exact corner
coordinates shown by the box editor. A positive-only review with whitening
marks represents a completed whole-card inspection. For older partial reviews
only explicitly completed corners qualify. Blank cards, prose-only notes and
uncertain corners never become background examples.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile

from PIL import Image

from pixel_training import EXPOSURE_SCHEMA, SCHEMA, box_valid, candidate_box, card_groups

CORNERS = ('TL', 'TR', 'BL', 'BR')
DEFAULT_LABELS = Path(__file__).resolve().parent / 'review_labels'
STATE = re.compile(rb'<script\b[^>]*\bid="pixel-annotation-state"[^>]*>(.*?)</script>', re.S)
# The review page includes many other image tags containing model-drawn boxes.
# Only the tagged original can qualify, and its encoded file hash must agree.
ORIGINAL = re.compile(rb'<img\b[^>]*\balt="Original saved corner sheet"[^>]*>', re.I)
SOURCE = re.compile(rb'\bsrc="data:image/(?:jpeg|jpg|png|webp);base64,([a-zA-Z0-9+/=]+)"', re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_records(directory, exposure_only=False):
    exposures = set()
    cards = {}
    if not directory.exists():
        return cards, exposures
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not path.name.endswith(('.json', '.json.gz')):
            continue
        if exposure_only and not path.name.endswith('.json'):
            continue
        raw = path.read_bytes()
        record = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
        if record.get('schema') == EXPOSURE_SCHEMA:
            keys = record['exposed_aliases']
            if not isinstance(keys, list) or not all(isinstance(x, str) for x in keys):
                raise ValueError(f'Invalid exposure record in {path}')
            exposures.update(keys)
        elif not exposure_only and record.get('schema') == SCHEMA:
            for card in record['cards']:
                key = card['source_sha256']
                if key in cards and review_fields(cards[key]) != review_fields(card):
                    raise ValueError(f'Conflicting label exports for {key} in {directory}')
                cards[key] = card
    return cards, exposures


def review_fields(card):
    return {k:card.get(k) for k in ('group_id', 'listing', 'split', 'annotations', 'notes', 'complete_corners')}


def read_review(path, label_cards):
    doc = path.read_bytes()
    match = STATE.search(doc)
    if not match:
        raise ValueError(f'{path.name}: use Save marked review with photos in the pixel box editor')
    state = json.loads(match[1])
    if state.get('schema') != SCHEMA:
        raise ValueError(f'{path.name}: unsupported review format')
    # Saved HTML omits the flag that exportData() adds on a JSON download.
    positive_only_editor = b'c.positive_only_review=Boolean(' in doc
    originals = {}
    for match in ORIGINAL.finditer(doc):
        data_uri = SOURCE.search(match[0])
        if not data_uri:
            raise ValueError(f'{path.name}: expected an embedded original corner sheet')
        data = base64.b64decode(data_uri[1], validate=True)
        originals[sha(data)] = data
    for embedded in state['cards']:
        key = embedded['source_sha256']
        if key not in originals:
            raise ValueError(f'{path.name}: original image missing or changed for card {embedded["number"]}')
        card = dict(embedded)
        labels = label_cards.get(key)
        if labels and not (embedded['annotations'] or embedded.get('notes') or embedded.get('complete_corners')):
            # Allow unmarked portable photos to be paired with the separate box export.
            card.update(review_fields(labels))
            card['positive_only_review'] = labels.get('positive_only_review', False)
        else:
            card['positive_only_review'] = (card.get('positive_only_review', False) or positive_only_editor
                or bool(labels and labels.get('positive_only_review') and review_fields(card) == review_fields(labels)))
        if card['split'] not in ('validation', 'development') or not card['listing'] or not card['group_id']:
            raise ValueError(f'{path.name}: missing group or split on card {card["number"]}')
        yield card, originals[key]


def corner_labels(card, side):
    complete = set(card.get('complete_corners', []))
    if complete.difference(CORNERS):
        raise ValueError('Invalid completed corner name')
    marks = card['annotations']
    for mark in marks:
        if (mark['corner'] not in CORNERS or mark['label'] not in ('whitening','not_whitening','uncertain')
                or not box_valid(mark['box'], side)):
            raise ValueError(f'Invalid mark in card {card["number"]}')
    clean = card.get('notes', '').strip().casefold().rstrip('.!').strip() == 'no whitening'
    has_chips = any(m['label'] == 'whitening' for m in marks)
    if clean and has_chips:
        raise ValueError(f'Card {card["number"]} has both a clean note and whitening marks')
    if clean or (has_chips and card.get('positive_only_review')):
        complete.update(CORNERS)
    unknown = {m['corner'] for m in marks if m['label'] == 'uncertain'}
    return {corner: [xyxy(m['box']) for m in marks if m['corner']==corner and m['label']=='whitening']
            for corner in CORNERS if corner in complete and corner not in unknown}


def xyxy(box):
    x,y,w,h = box
    return [x,y,x+w,y+h]


def baseline_regions(card):
    result = card['result']
    model = result.get('whitening_classification')
    if not model or 'decisions' not in model:
        return None
    candidates = {c['id']: c for c in result['candidates'] + result['excluded_candidates']}
    regions = {corner:[] for corner in CORNERS}
    for decision in model['decisions']:
        if decision['classification'] == 'likely_whitening':
            c = candidates[decision['id']]
            regions[c['corner']].append(xyxy(candidate_box(c,decision)))
    return regions


def write_dataset(cards, originals, exposures, out, history=()):
    (out/'images').mkdir()
    images = []
    skipped = 0
    present = {id(c) for c in cards}
    for keys, related in card_groups([*history, *cards]):
        members = [c for c in related if id(c) in present]
        if not members:
            continue
        stable_key = min(k for k in keys if k.startswith('group:'))
        splits = {c['split'] for c in members}
        used = bool(exposures.intersection(keys)) or any(
            c.get('previously_reviewed') or (id(c) not in present and c['split'] == 'development')
            for c in related)
        if len(splits)>1 and not used:
            raise ValueError(f'Conflicting development and reserved assignments for {stable_key}')
        if splits == {'validation'} and not used:
            split = 'test'
        else:
            # Independent of input order and dataset size. Old groups retain
            # their roles as unrelated new reviews arrive.
            split = 'validation' if int(sha(stable_key.encode())[:8],16)%5 == 0 else 'train'
        for card in sorted(members,key=lambda c:c['source_sha256']):
            key = card['source_sha256']
            width, height = card['result']['measurement_image_size']
            side = (width-10)//2
            if side<32 or width!=side*2+10 or height!=side*2+50:
                raise ValueError(f'Invalid corner sheet layout for {key}')
            labels = corner_labels(card,side)
            skipped += 4-len(labels)
            if not labels:
                continue
            baseline = baseline_regions(card)
            with Image.open(io.BytesIO(originals[key])) as source:
                if source.size != (width,height):
                    raise ValueError(f'Review measurements and photo dimensions differ for {key}')
                sheet = source.convert('RGB')
            for corner,boxes in labels.items():
                x = side+10 if corner[1]=='R' else 0
                y = side+50 if corner[0]=='B' else 20
                rel = f'images/{key}-{corner}.png'
                patch = sheet.crop((x,y,x+side,y+side))
                patch.save(out/rel)
                row = dict(file=rel,image_sha256=sha((out/rel).read_bytes()),group_id=stable_key,
                           aliases=sorted(keys),split=split,boxes=boxes,corner=corner,source_sha256=key)
                if baseline is not None:
                    row['baseline_boxes'] = baseline[corner]
                images.append(row)
    if not images:
        raise ValueError('No fully reviewed corners. Save green boxes or write No whitening for a clean card.')
    images.sort(key=lambda row:row['file'])
    counts = {s: dict(corners=sum(r['split']==s for r in images),
                      groups=len({r['group_id'] for r in images if r['split']==s}),
                      chips=sum(len(r['boxes']) for r in images if r['split']==s)) for s in ('train','validation','test')}
    manifest = dict(schema='whitening-detector-v1', images=images, counts=counts, skipped_corners=skipped)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',encoding='utf-8')
    return manifest


def build(reviews, labels, out, exposure_dir=None):
    files = sorted({f.resolve() for p in reviews for f in (p.glob('*.html') if p.is_dir() else [p])})
    if not files:
        raise ValueError('Choose the saved HTML review(s) containing your boxes and original photos')
    label_cards, exposures = load_records(labels)
    if exposure_dir:
        exposures.update(load_records(exposure_dir, exposure_only=True)[1])
    cards = {}
    originals = {}
    for path in files:
        for card, data in read_review(path,label_cards):
            key = card['source_sha256']
            if key in cards and review_fields(cards[key]) != review_fields(card):
                raise ValueError(f'Conflicting saved reviews of {key}; keep the latest corrected copy')
            cards[key] = card
            originals[key] = data
    if out.exists():
        raise ValueError(f'Choose a new dataset output folder: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.detector-data-',dir=out.parent))
    try:
        # Earlier group assignments remain known even for absent photos.
        result = write_dataset(list(cards.values()), originals, exposures,temp,label_cards.values())
        temp.rename(out)
        return result
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', nargs='+',type=Path,required=True,help='Saved review HTML files or their folder')
    parser.add_argument('--labels',type=Path,default=DEFAULT_LABELS,help='Folder of optional label exports and pixel exposure records')
    parser.add_argument('--exposure-dir',type=Path,help='Additional detector validation exposure records')
    parser.add_argument('--out',type=Path,required=True,help='New dataset folder')
    args=parser.parse_args()
    try:
        manifest = build(args.reviews,args.labels,args.out,args.exposure_dir)
    except (OSError,KeyError,ValueError,TypeError) as exc:
        parser.exit(1,f'Could not prepare detector data. {exc}\n')
    print(json.dumps(manifest['counts'],indent=2))
    print(f"Skipped {manifest['skipped_corners']} uncertain or unreviewed corners. Manifest: {args.out/'manifest.json'}")


if __name__=='__main__':
    main()
