"""Portable, offline human review for card-run artifacts. No API calls."""
import argparse
import base64
import hashlib
import json
import mimetypes
from pathlib import Path


def local_asset(out, value):
    """Relocate CI paths using the audit root, never read outside the run."""
    if not value:
        return None
    value = str(value).replace('\\', '/')
    if '/_audit/' in value:
        value = '_audit/' + value.split('/_audit/', 1)[1]
    p = (out / value).resolve()
    if not p.is_relative_to(out.resolve()):
        return None
    return p if p.is_file() else None


def build_batch(out, state, production=False, embed=True):
    cards = []
    for row in state['records']:
        if production and not row.get('accepted'):
            continue
        packet = row.get('back_packet', {})
        assets = []
        contexts = packet.get('context') or []
        if isinstance(contexts, str):
            contexts = [contexts]
        assets.extend((p, f'Original listing photo {i}') for i, p in enumerate(contexts, 1))
        if row.get('source'):
            assets.append(('_audit/' + row['source'], 'Original front photo'))
        for key, label in [('sheet', 'Edge and corner sheet'), ('tips', 'Corner close-ups')]:
            if packet.get(key):
                assets.append((packet[key], label))
        for name, path in row.get('overlays', {}).items():
            assets.append(('_audit/' + path, 'Centering: ' + name))
        manifest = local_asset(out, packet.get('manifest'))
        if manifest:
            for photo in json.loads(manifest.read_text(encoding='utf-8-sig')).get('photos', []):
                for key, label in [('original_file', 'Original listing photo'), ('boundary_overlay', 'Detected back outline')]:
                    if photo.get(key):
                        assets.append((str(manifest.parent / photo[key]), label))
        images, seen = [], set()
        for value, label in assets:
            p = local_asset(out, value)
            if not p or p in seen:
                continue
            mime = mimetypes.guess_type(p.name)[0]
            if mime not in {'image/jpeg', 'image/png', 'image/webp'}:
                continue
            seen.add(p)
            raw = p.read_bytes()
            images.append({'label': label, 'file': p.relative_to(out.resolve()).as_posix(),
                           'sha256': hashlib.sha256(raw).hexdigest(),
                           'src': 'data:' + mime + ';base64,' + base64.b64encode(raw).decode() if embed else p.relative_to(out.resolve()).as_posix()})
        whitening = row.get('whitening', {})
        url = str(row.get('url', ''))
        cards.append({'id': row['item_id'], 'number': len(cards)+1, 'title': row['title'],
                      'url': url if url.startswith(('https://', 'http://')) else '',
                      'pipeline_decision': {'status': row.get('status', 'undetermined'), 'accepted': row.get('accepted', False),
                                            'reason': row.get('reason', ''), 'whitening': whitening.get('decision', 'not_run'),
                                            'centering': row.get('centering_gate'), 'measurements': row.get('combined', {}).get('axes', {}),
                                            'evidence': [{k: s.get(k, '') for k in ['decision', 'location', 'evidence']} for s in whitening.get('stages', [])],
                                            'input_mode': packet.get('input_mode', ''), 'error': row.get('error') or whitening.get('error')},
                      'timings': row.get('timings', {}), 'images': images})
    if len({c['id'] for c in cards}) != len(cards):
        raise ValueError('Duplicate listing IDs in review')
    identity = {'started_utc': state.get('started_utc'), 'pipeline': state['settings']['pipeline'],
                'cards': [{**c, 'images': [{k: v for k, v in im.items() if k != 'src'} for im in c['images']]} for c in cards]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {'schema_version': 1, 'batch_id': out.name, 'batch_sha256': fingerprint,
            'pipeline': state['settings']['pipeline'], 'run_state': state.get('state'),
            'portable': embed, 'cards': cards}


def write_review(out, state, production=False, embed=True):
    out = Path(out).resolve()
    batch = build_batch(out, state, production, embed)
    template = Path(__file__).with_name('card_review.html').read_text(encoding='utf-8')
    data = json.dumps(batch, ensure_ascii=True, separators=(',', ':')).replace('<', '\\u003c')
    (out / 'review.html').write_text(template.replace('__BATCH_DATA__', data), encoding='utf-8')
    (out / 'REVIEW_README.txt').write_text('Open review.html in your browser. Completed runs embed all available photos. Mark each decision Correct, Incorrect or Unsure; add a corrected decision and notes if useful. Save reviewed HTML shares photos and answers together. Export review JSON is the compact file for analysis or chat models. Import JSON resumes a saved review. Download backups regularly; browser autosave is a convenience, not a portable backup. No answers are sent to a server.\n', encoding='utf-8')
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder')
    args = parser.parse_args()
    out = Path(args.run_folder)
    state = json.loads((out / '_audit/run.json').read_text(encoding='utf-8-sig'))
    batch = write_review(out, state, state['settings'].get('output_mode') == 'production')
    print(json.dumps({'cards': len(batch['cards']), 'images': sum(len(c['images']) for c in batch['cards']), 'review': str(out / 'review.html')}))
