"""Prepare sheets from human-marked corners in a portable reviewed HTML file.

Reads embedded JSON only; never executes HTML. Model calls require --run-whitening.
"""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from card_utils import edge_corner_sheet, corner_closeup_sheet
from whitening_geometry import warp


def read_document(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    def read(name):
        match = re.search(r'<script\b[^>]*\bid=["\']' + name + r'["\'][^>]*>(.*?)</script\s*>', text, re.S | re.I)
        if not match:
            raise ValueError('Missing embedded ' + name)
        return json.loads(match.group(1))
    return read('batch-data'), read('saved-review')


def validate_corners(value, width, height):
    if value.get('width') != width or value.get('height') != height:
        raise ValueError('Original image dimensions do not match')
    points = np.asarray(value.get('points'), dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
        raise ValueError('Expected four normalized corners inside the photo')
    q = points.astype(np.float32)
    if not cv2.isContourConvex(q) or cv2.contourArea(q, oriented=True) < .01:
        raise ValueError('Corners must be clockwise TL, TR, BR, BL and surround the card')
    if not (q[:2, 1].mean() < q[2:, 1].mean() and q[[0, 3], 0].mean() < q[[1, 2], 0].mean()):
        raise ValueError('Corner order must be TL, TR, BR, BL')
    return (q * [width-1, height-1]).astype(np.float32)


def selected_images(batch, review):
    if not review or review.get('batch_id') != batch.get('batch_id') or review.get('batch_sha256') != batch.get('batch_sha256'):
        raise ValueError('Saved review does not match this batch')
    cards = {c['id']: c for c in batch['cards']}
    rows = review['cards']
    if len(cards) != len(batch['cards']) or len(rows) != len(cards) or {r['id'] for r in rows} != set(cards):
        raise ValueError('Duplicate, missing or unexpected cards')
    selected = []
    for row in rows:
        correction = row.get('human_review', {}).get('manual_corners')
        if correction is None:
            continue
        card = cards[row['id']]
        matches = [im for im in card['images'] if im['file'] == correction.get('image_file') and im['sha256'] == correction.get('image_sha256') and im['label'].startswith('Original')]
        if len(matches) != 1:
            raise ValueError('Correction does not identify one original photo')
        im = matches[0]
        match = re.fullmatch(r'data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/=\s]+)', im['src'])
        if not match:
            raise ValueError('Use a completed portable HTML review with embedded photos')
        raw = base64.b64decode(match[2], validate=True)
        if hashlib.sha256(raw).hexdigest() != correction['image_sha256']:
            raise ValueError('Original image hash does not match')
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Unreadable original image')
        h, w = image.shape[:2]
        quad = validate_corners(correction, w, h)
        selected.append((card, row, correction, raw, match[1], image, quad))
    if not selected:
        raise ValueError('No saved manual corner corrections')
    return selected


def prepare(path, out, *, classifier=None, bank=None):
    from card_review import write_review
    batch, review = read_document(path)
    selected = selected_images(batch, review)  # Validate everything before writing.
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never replace an earlier run.
    state = {'settings': {'pipeline': 'api'}, 'state': 'manual_preparation_complete', 'records': []}
    for number, (card, row, correction, raw, extension, image, quad) in enumerate(selected, 1):
        folder = out / ('card_' + str(number))
        folder.mkdir()
        original = folder / ('original.' + extension)
        original.write_bytes(raw)
        normalized = warp(image, quad)
        overlay = image.copy()
        cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (0, 255, 0), 3)
        assets = {'normalized.jpg': normalized, 'sheet.jpg': edge_corner_sheet(normalized, 'back'),
                  'tips.png': corner_closeup_sheet(normalized, 'back'), 'outline.jpg': overlay}
        for name, pixels in assets.items():
            if not cv2.imwrite(str(folder / name), pixels):
                raise OSError('Failed to save ' + name)
        packet = {'ready': True, 'card_id': card['id'], 'context': str(original), 'sheet': str(folder/'sheet.jpg'),
                  'tips': str(folder/'tips.png'), 'input_mode': 'manual_crops', 'manifest': str(folder/'manifest.json')}
        (folder/'manifest.json').write_text(json.dumps({'photos': [{'original_file': original.name, 'boundary_overlay': 'outline.jpg'}]}), encoding='utf-8')
        record = {'item_id': card['id'], 'title': card['title'], 'url': card['url'], 'status': 'undetermined',
                  'accepted': False, 'reason': 'manual_crops_prepared_not_reassessed', 'whitening': {'decision': 'not_run'},
                  'back_packet': packet, 'manual_corners': correction, 'source_human_review': row['human_review'],
                  'previous_pipeline_decision': card['pipeline_decision']}
        state['records'].append(record)
    if classifier is not None:
        import luna_whitening as luna
        if not bank:
            raise ValueError('A reference bank is required for reassessment')
        # Validate all references before the first paid request.
        refs = [luna.references(bank, r['item_id']) for r in state['records']]
    (out/'_audit').mkdir()
    def save():
        (out/'_audit'/'run.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        write_review(out, state)
    save()
    if classifier is not None:
        for record, references in zip(state['records'], refs):
            packet = record['back_packet']
            record['reason'] = 'manual_reassessment_in_flight'
            save()
            try:
                result = classifier(packet['context'], packet['sheet'], packet['tips'], references)
                record['whitening'] = result
                decision = result['decision']
                record.update(status={'clean': 'pass', 'not_clean': 'reject'}.get(decision, 'undetermined'),
                              accepted=decision == 'clean', reason='manual_reassessment_' + decision)
            except Exception:
                record.update(reason='manual_reassessment_error', whitening={'decision': 'needs_review'})
                save()
                raise
            save()
        state['state'] = 'manual_reassessment_complete'
        save()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reviewed_html')
    parser.add_argument('--out', required=True, help='New output folder (must not exist)')
    parser.add_argument('--run-whitening', action='store_true', help='Explicitly enable paid whitening calls after preparing corrected sheets')
    parser.add_argument('--references', default='luna_references.json')
    parser.add_argument('--env-file', default='.env')
    args = parser.parse_args()
    kwargs = {}
    if args.run_whitening:
        import luna_whitening as luna
        from card_runner import load_bank
        from dotenv import load_dotenv
        load_dotenv(args.env_file)
        kwargs = {'classifier': luna.classify, 'bank': load_bank(args.references)}
    result = prepare(args.reviewed_html, args.out, **kwargs)
    print(json.dumps({'prepared_cards': len(result['records']), 'whitening_requested': args.run_whitening,
                      'review': str(Path(args.out)/'review.html')}))
