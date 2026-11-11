"""Offline pixel-detector review from a portable reviewed HTML. No API calls."""
import argparse
import base64
import html
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from pixel_whitening import inspect
from review_corners import read_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review')
    parser.add_argument('--out', required=True)
    parser.add_argument('--surround-threshold', type=float)
    parser.add_argument('--radius-fraction', type=float, default=.02)
    parser.add_argument('--geometry', choices=['ring','triangle','square','circle'], default='ring')
    parser.add_argument('--no-large-exception', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    batch, review = read_document(args.review)
    answers = {c['id']: c.get('human_review', {}) for c in (review or {}).get('cards', [])}
    results, panels = [], []
    cv2.setNumThreads(2)
    for card in batch['cards']:
        source = next((im for im in card['images'] if im['label']=='Corner close-ups'), None)
        if source is None:
            results.append(dict(number=card['number'], status='no_saved_corner_sheet'))
            panels.append(f'<section><h2>Card {card["number"]}</h2><p>No saved corner close-up sheet; skipped.</p></section>')
            continue
        raw = base64.b64decode(source['src'].split(',', 1)[1], validate=True)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        result, overlay = inspect(image, args.surround_threshold, args.radius_fraction, args.geometry,
                                  not args.no_large_exception)
        if args.surround_threshold is not None:
            result['band_calibration'] = 'Not used; local blue geometry filter replaces boundary filter'
            result['surrounding_blue_settings'] = dict(threshold=args.surround_threshold,
                radius_fraction=args.radius_fraction, geometry=args.geometry,
                minimum_sector_blue_fraction=.35 if args.geometry=='ring' else None,
                large_exception_area_at_500px=None if args.no_large_exception else 100)
        result.update(number=card['number'], listing=card['id'], human_review=answers.get(card['id'], {}),
                      prior_decision=card['pipeline_decision']['whitening'],
                      source_file=source['file'], source_sha256=hashlib.sha256(raw).hexdigest())
        results.append(result)
        filename = f"card-{card['number']}.png"
        if not cv2.imwrite(str(out/filename), overlay):
            raise OSError('Could not write overlay')
        encoded = base64.b64encode((out/filename).read_bytes()).decode()
        details = html.escape(json.dumps(result, indent=2))
        panels.append(f'<section><h2>Card {card["number"]}</h2><p>{result["candidate_count"]} light-spot candidates</p>'
                      '<div class="comparison"><figure><figcaption>Saved corner sheet</figcaption>'
                      f'<img src="{html.escape(source["src"], quote=True)}" alt="Original saved corner sheet"></figure>'
                      '<figure><figcaption>Candidate marks</figcaption>'
                      f'<img src="data:image/png;base64,{encoded}" alt="Candidate light spots on the same corners"></figure></div>'
                      f'<details><summary>Measurements and your prior review</summary><pre>{details}</pre></details></section>')
        print(card['number'], result['candidate_count'], flush=True)
    (out/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    page = ('<!doctype html><meta charset="utf-8"><title>Experimental pixel inspection</title>'
            '<style>body{font:17px system-ui;max-width:1100px;margin:30px auto;padding:16px;background:#fafafa}'
            'section{border-top:1px solid #ccc;padding:20px 0}img{max-width:100%;max-height:850px}'
            '.comparison{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}figcaption{margin:8px 0}'
            '@media(max-width:750px){.comparison{grid-template-columns:1fr}}'
            'pre{white-space:pre-wrap}</style><h1>Four-corner pixel inspection</h1>'
            '<p>These are the exact corner close-ups saved by the sheet-making pipeline. Red boxes identify '
            'local light spots near blue pixels. Cyan shows the inner boundary measured separately on each corner: '
            '2.4% of crop width inward. There is no outer cutoff. '
            'Larger interior marks can be retained by exception. Candidate numbers match the prior unfiltered review. '
            'A spot may be damage, glare, dust, or background. '
            'Numbers link to measurements below each card. All findings require review; an unmarked corner '
            'has not been certified clean.</p>'
            + ''.join(panels))
    (out/'review.html').write_text(page, encoding='utf-8')
    if args.surround_threshold is not None:
        page = page.replace('Cyan shows the inner boundary measured separately on each corner: '
                            '2.4% of crop width inward. There is no outer cutoff.',
                            f'No outline filter is used. Small spots are excluded when at least '
                            f'{args.surround_threshold:.0%} of their perimeter ring is blue, with at least '
                            f'35% blue in each of eight directions. Ring width: {args.radius_fraction:.1%} '
                            'of the corner crop. Incomplete rings are retained. '
                            'Green boxes show excluded spots; red boxes show retained candidates.')
        (out/'review.html').write_text(page, encoding='utf-8')
        if args.geometry in ('triangle','square','circle'):
            start=page.index('No outline filter is used.')
            end=page.index('Larger interior marks',start)
            page=page[:start]+(
                f'Cyan triangles have their tip at each candidate pixel centroid and their base toward '
                f'the corner (TR 45°, TL 135°, BL 225°, BR 315°; standard math angles). '
                f'Tip-to-base distance: {args.radius_fraction:.1%} of crop width. '
                f'Candidates with at least {args.surround_threshold:.0%} blue inside the triangle are excluded '
                'unless protected by the large-mark exception. Candidate pixels are included in the measurement. '
                'Triangles crossing the crop boundary are retained. No ring or directional-sector rule is used. '
                'Green boxes: excluded; red boxes: retained. ')+page[end:]
            (out/'review.html').write_text(page,encoding='utf-8')
        if args.geometry in ('square','circle'):
            start=page.index('Cyan triangles')
            end=page.index('Larger interior marks',start)
            label='side length' if args.geometry=='square' else 'diameter'
            page=page[:start]+(f'Cyan {args.geometry}s are centered on each candidate pixel centroid. '
                f'{label}: {args.radius_fraction:.1%} of crop width. '
                f'Exclude when at least {args.surround_threshold:.0%} of pixels inside the shape are blue. '
                'Candidate pixels count in the measurement. Incomplete shapes are retained. '
                'No outline or directional-sector filter is used. Green: excluded; red: retained. ')+page[end:]
        if args.no_large_exception:
            page=page.replace('unless protected by the large-mark exception.', '(large-mark exception disabled).')
            page=page.replace('Larger interior marks can be retained by exception.', 'Large-mark exception disabled.')
        (out/'review.html').write_text(page,encoding='utf-8')


if __name__ == '__main__':
    main()
