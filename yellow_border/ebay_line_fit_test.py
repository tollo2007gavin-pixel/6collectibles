"""Run in VS Code: eBay listing fronts -> one combined local yellow-border image.

Only eBay authentication/search/detail/image requests are made. No OpenAI calls.
See EBAY_LINE_FIT_README.md for setup, output meanings and offline testing.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlparse

# ========================= EDIT THESE SETTINGS =========================
TOTAL_CARDS_ANALYZED = 45      # Total listing-front images processed, across all CSV rows.
MAX_LISTINGS_PER_CARD = 3      # Maximum fronts for each active card/search_query in cards.csv.
MAX_CANDIDATES_PER_CARD = 60   # Search/detail budget; non-NM listings do not count as analyzed.
LISTING_AGE_HOURS = None       # None = any age; e.g. 24 = listings started in the last day.
SHUFFLE_CARDS = True          # Spread small batches across the active CSV rows.
RANDOM_SEED = None            # Set an integer for repeatable card-query order.
EXCLUDE_PREVIOUS_BATCH = True # Skip listing IDs from the immediately preceding batch.
MARKETPLACE_ID = 'EBAY_US'
CARD_RARITY_FILTERS = {'39': 'Ultra Rare', '28': 'Ultra Rare'}  # Ditto GG22/GG70 and Turtwig GG31/GG70.
REQUEST_TIMEOUT = (15, 60)    # Connection/read timeout, seconds.
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MEASUREMENT_FRAME = 'refit_when_supported' # Rebuilt frame where supported; initial frame fallback.
PERSPECTIVE_LIMIT = 0.20      # Measurement-quality limit; not a centering-ratio gate.
ANGULAR_REVIEW_THRESHOLD = 0.15 # Above this, mark the proposed +/-1 percentage-point allowance.
# =====================================================================

PROJECT_DIR = Path(__file__).resolve().parent
CARDS_CSV = PROJECT_DIR / 'data' / 'cards.csv'
OUTPUT_ROOT = PROJECT_DIR / 'centering_output' / 'ebay_line_fit'
OWNER = 'proj001-yellow-ebay-line-fit-v1'
MANAGED_FOLDERS = ('Combined', 'Forced_curve', 'adaptive', 'downloads', 'details')
METHOD_FOLDERS = {'adaptive': 'adaptive', 'forced': 'Forced_curve'}


def load_cards(path=CARDS_CSV):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'id', 'search_query', 'active'} <= set(reader.fieldnames or []):
            raise ValueError('cards.csv needs id, search_query and active columns')
        cards = [r for r in reader if r['active'].strip() == '1' and r['search_query'].strip()]
    if not cards:
        raise ValueError('No active card queries in cards.csv')
    return cards


def is_near_mint_ungraded(details):
    # Use structured seller condition, never a title mentioning "PSA 10" or "NM".
    if str(details.get('conditionId', '')) != '4000':
        return False
    for descriptor in details.get('conditionDescriptors', []) or []:
        if str(descriptor.get('name', '')).strip().casefold() != 'card condition':
            continue
        return any(str(v.get('content', '')).strip().casefold() == 'near mint or better'
                   for v in descriptor.get('values', []) or [])
    return False


class EbayClient:
    def __init__(self, project_dir=PROJECT_DIR):
        import requests
        from dotenv import load_dotenv
        load_dotenv(Path(project_dir) / '.env')
        client_id, client_secret = os.getenv('EBAY_CLIENT_ID'), os.getenv('EBAY_CLIENT_SECRET')
        if not client_id or not client_secret:
            raise ValueError('Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET in Proj001/.env')
        self.api = requests.Session()
        self.images = requests.Session()  # Image downloads never receive the bearer token.
        response = self.api.post('https://api.ebay.com/identity/v1/oauth2/token',
            auth=(client_id, client_secret), data={'grant_type': 'client_credentials',
            'scope': 'https://api.ebay.com/oauth/api_scope'}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        self.api.headers.update({'Authorization': 'Bearer '+response.json()['access_token'],
                                 'X-EBAY-C-MARKETPLACE-ID': MARKETPLACE_ID, 'Accept-Language': 'en-US'})

    def search(self, query, *, offset, limit, age_hours, card_id=None):
        filters = 'conditionIds:{4000}'
        if age_hours is not None:
            cutoff = (datetime.now(timezone.utc)-timedelta(hours=age_hours)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
            filters += f',itemStartDate:[{cutoff}..]'
        params = {
            'q': query, 'category_ids': '183454', 'limit': limit, 'offset': offset,
            'filter': filters, 'sort': 'newlyListed'}
        rarity = CARD_RARITY_FILTERS.get(str(card_id))
        if rarity:
            params['aspect_filter'] = f'categoryId:183454,Rarity:{{{rarity}}}'
        response = self.api.get('https://api.ebay.com/buy/browse/v1/item_summary/search',
                                params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def details(self, item_id):
        response = self.api.get('https://api.ebay.com/buy/browse/v1/item/'+quote(item_id, safe=''), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def download(self, url):
        if urlparse(url).scheme != 'https':
            raise ValueError('Listing image must use HTTPS')
        with self.images.get(url, timeout=REQUEST_TIMEOUT, stream=True) as response:
            response.raise_for_status()
            if int(response.headers.get('Content-Length', 0)) > MAX_IMAGE_BYTES:
                raise ValueError('Main image exceeds download size limit')
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > MAX_IMAGE_BYTES:
                    raise ValueError('Main image exceeds download size limit')
        return bytes(content)

    def close(self):
        self.api.close(); self.images.close()


def linked(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


class BatchOutput:
    """Own only this runner's dedicated folders; reject links before clearing."""
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.locked = False
        self.previous_ids = set()

    def __enter__(self):
        root = self.root
        for path in (root, *root.parents):
            if linked(path):
                raise ValueError(f'Refusing an output path through a link/junction: {path}')
        if root.exists() and any(root.iterdir()):
            marker = root / '.owner'
            if not marker.is_file() or marker.read_text() != OWNER:
                raise ValueError(f'Output folder is not owned by this runner: {root}')
        root.mkdir(parents=True, exist_ok=True)
        # Exclusive lock prevents two VS Code runs from clearing each other's files.
        try:
            fd = os.open(root / '.running', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f'Another run is active, or a stopped run left {root / ".running"}. See the README.') from None
        os.close(fd); self.locked = True
        try:
            (root / '.owner').write_text(OWNER)
            manifest = root / 'batch.json'
            if manifest.is_file():
                self.previous_ids = {r['item_id'] for r in json.loads(manifest.read_text(encoding='utf-8')).get('records', [])}
            # Validate every deletion target and descendant before deleting anything.
            for name in MANAGED_FOLDERS:
                target = root / name
                if target.resolve().parent != root.resolve() or linked(target):
                    raise ValueError(f'Unsafe cleanup target: {target}')
                if target.exists():
                    if not target.is_dir(): raise ValueError(f'Expected folder: {target}')
                    if any(linked(p) for p in target.rglob('*')):
                        raise ValueError(f'Cleanup folder contains a link/junction: {target}')
            for name in MANAGED_FOLDERS:
                target = root / name
                if target.exists(): shutil.rmtree(target)
                if name not in ('Forced_curve', 'adaptive'):
                    target.mkdir()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self.locked:
            (self.root / '.running').unlink(missing_ok=True)
            self.locked = False


def json_default(value):
    if hasattr(value, 'tolist'): return value.tolist()
    if hasattr(value, 'item'): return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=json_default, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def checkpoint(root, batch):
    write_json(root / 'batch.json', batch)
    fields = ['card_id', 'query', 'item_id', 'title', 'url', 'technique', 'status',
              'left_pct', 'right_pct', 'top_pct', 'bottom_pct', 'overlay', 'reasons',
              'candidate_frame_left_pct','candidate_frame_top_pct',
              'lr_frame_change_pp','tb_frame_change_pp',
              'lr_resampling_shift_pp','tb_resampling_shift_pp','frame_diagnostics_policy',
              'measurement_category','lr_angular_allowance_pp','tb_angular_allowance_pp']
    with (root / 'measurements.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in batch['records']:
            for mode, result in {'combined': row['combined']}.items():
                record = {k: row.get(k, '') for k in fields[:5]}
                record.update(technique=mode, status=result['status'], overlay=row['overlays'][mode])
                for axis, sides in [('lr', ('left', 'right')), ('tb', ('top', 'bottom'))]:
                    a = result['axes'][axis]
                    if a['resolved']:
                        record[sides[0]+'_pct'] = a['percentage']; record[sides[1]+'_pct'] = 100-a['percentage']
                        diagnostic=a.get('frame_diagnostics',{})
                        if diagnostic.get('candidate_available'):
                            record['candidate_frame_'+sides[0]+'_pct']=diagnostic['candidate_percentage']
                            record[axis+'_frame_change_pp']=diagnostic['frame_difference_pp']
                        if 'max_abs_candidate_resampling_shift_pp' in diagnostic:
                            record[axis+'_resampling_shift_pp']=diagnostic['max_abs_candidate_resampling_shift_pp']
                record['frame_diagnostics_policy']='advisory_only; '+', '.join(axis+'='+a.get('frame_diagnostics',{}).get('active_frame','initial') for axis,a in result['axes'].items())
                record['reasons'] = '; '.join(dict.fromkeys(r for a in result['axes'].values() for r in a['reasons']))
                policy=result.get('measurement_policy',{})
                record['measurement_category']=policy.get('category','')
                for axis in ('lr','tb'):
                    record[axis+'_angular_allowance_pp']=policy.get('axis_allowance_pp',{}).get(axis)
                writer.writerow(record)
    rows = []
    for row in batch['records']:
        link = row.get('url', '')
        title = html.escape(row['title'])
        heading = f'<a href="{html.escape(link, quote=True)}">{title}</a>' if link.startswith('https://') else title
        panels = f'<div><a href="{row["overlays"]["combined"]}"><img loading="lazy" src="{row["overlays"]["combined"]}"></a></div>'
        rows.append(f'<article><h2>{heading}</h2><p>{html.escape(row["query"])} · {html.escape(row["item_id"])}</p><div class="pair">{panels}</div></article>')
    page = '<!doctype html><meta charset="utf-8"><title>eBay combined yellow-border fits</title><style>body{font:16px system-ui;background:#17202a;color:white;margin:24px}a{color:#9de}img{width:100%;max-width:1100px}article{margin:40px 0}</style><h1>eBay combined yellow-border fits</h1><p>One image per listing. Forced-curve lines where available, adaptive fallback. Each margin ratio averages the two qualified method percentages, or uses the single qualified method. Adaptive may select straight or curved lines. Unresolved measurements stay blank. Green exterior, red interior. Seller-listed near mint is not a grading verdict.</p><p><a href="measurements.csv">Combined measurements</a> · <a href="batch.json">Both methods and combined results</a></p>'
    page+='<p>Rebuilt frames are used where supported, with initial-frame fallback. The default perspective limit is 0.20; measurements above 0.15 are marked pass/review with a proposed +/-1 percentage-point angular allowance. This is an operating allowance, not a calibrated confidence interval. Pass means measurement completed, not that a card meets a centering selection gate. No centering selection gate is active. CSV resizing shifts remain advisory.</p>'
    (root / 'review.html').write_text(page+(''.join(rows) or '<p>No images processed in this batch.</p>'), encoding='utf-8')


def process_front(content, metadata, root, analyzer=None):
    import numpy as np
    import cv2
    from PIL import Image, ImageOps
    from yellow_batch_engine.compare import compare, render, unresolved, combine_methods
    from yellow_batch_engine.batch_policy import analyze_front,describe_measurement
    with Image.open(io.BytesIO(content)) as opened:
        if opened.width * opened.height > MAX_IMAGE_PIXELS:
            raise ValueError('Main image exceeds pixel limit')
        original = ImageOps.exif_transpose(opened).convert('RGB')
    stem = re.sub(r'[^A-Za-z0-9_-]', '_', str(metadata['card_id']))[:35]+'_'+hashlib.sha256(metadata['item_id'].encode()).hexdigest()[:16]
    source = root / 'downloads' / (stem+'.png')
    original.save(source)  # Lossless working copy of the one downloaded main image.
    image = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
    try:
        analysis = analyzer(image) if analyzer else analyze_front(image,measurement_frame=MEASUREMENT_FRAME,perspective_limit=PERSPECTIVE_LIMIT)
    except Exception as error:
        # A failure must remain visible, rather than silently disappearing from the batch.
        analysis = {'error': str(error), 'methods': {m: dict(status='Analysis error', axes=unresolved(str(error)),
            paths={k: [None]*4 for k in ('outer','inner')}, trials=[]) for m in METHOD_FOLDERS}}
    combined = combine_methods(analysis['methods'])
    combined['measurement_policy']=describe_measurement(analysis,combined,review_threshold=ANGULAR_REVIEW_THRESHOLD)
    analysis['combined'] = combined
    relative = f'Combined/{stem}.jpg'
    render(image, combined, root / relative, title=metadata['title'], mode='combined')
    overlays = {'combined': relative}
    detail = f'details/{stem}.json'
    write_json(root / detail, dict(metadata=metadata, source_sha256=hashlib.sha256(content).hexdigest(), analysis=analysis))
    return dict(metadata, overlays=overlays, details=detail, source=f'downloads/{stem}.png',
                image_gate=analysis.get('image_gate'),
                combined={k: combined[k] for k in ('status','axes','path_sources','image_gate','measurement_policy') if k in combined}, methods={
        mode: {k: value[k] for k in ('status','axes','image_gate') if k in value} for mode,value in analysis['methods'].items()})


def run_batch(cards, client, output, *, total, per_card, candidates_per_card, age_hours,
              exclude_previous=True, processor=process_front):
    batch = dict(started_utc=datetime.now(timezone.utc).isoformat(), state='running', records=[], errors=[],
        settings=dict(total_cards_analyzed=total, max_listings_per_card=per_card,
                      max_candidates_per_card=candidates_per_card, listing_age_hours=age_hours,
                      exclude_previous_batch=exclude_previous,measurement_frame=MEASUREMENT_FRAME,
                      perspective_limit=PERSPECTIVE_LIMIT,angular_review_threshold=ANGULAR_REVIEW_THRESHOLD,
                      centering_selection_gate_applied=False), queries_attempted=0)
    seen = set(output.previous_ids) if exclude_previous else set()
    checkpoint(output.root, batch)
    try:
        for card in cards:
            if len(batch['records']) >= total: break
            query = card['search_query'].strip(); count = 0; offset = 0
            batch['queries_attempted'] += 1
            print(f"Searching {query} ({len(batch['records'])}/{total} fronts processed)", flush=True)
            while offset < candidates_per_card and count < per_card and len(batch['records']) < total:
                limit = min(200, candidates_per_card-offset)
                try:
                    page = client.search(query, offset=offset, limit=limit, age_hours=age_hours, card_id=card['id'])
                except Exception as error:
                    batch['errors'].append(dict(query=query, stage='search', error=str(error)))
                    print(f'  Search failed: {error}', flush=True); break
                items = page.get('itemSummaries', [])
                if not items: break
                for item in items[:limit]:
                    if count >= per_card or len(batch['records']) >= total: break
                    item_id = item.get('itemId')
                    if not item_id or item_id in seen: continue
                    seen.add(item_id)
                    try:
                        detail = client.details(item_id)
                        if not is_near_mint_ungraded(detail): continue
                        # Only the main image. additionalImages is intentionally never read.
                        image_url = (detail.get('image') or {}).get('imageUrl') or (item.get('image') or {}).get('imageUrl')
                        if not image_url: continue
                        metadata = dict(card_id=card['id'], query=query, item_id=item_id,
                            title=detail.get('title') or item.get('title') or query,
                            url=detail.get('itemWebUrl') or item.get('itemWebUrl') or '',
                            condition='Ungraded / Near mint or better', image_url=image_url)
                        print(f"  Fitting {metadata['title']}", flush=True)
                        row = processor(client.download(image_url), metadata, output.root)
                        batch['records'].append(row); count += 1
                        print('  '+', '.join(f'{m}: {r["status"]}' for m,r in row['methods'].items()), flush=True)
                        checkpoint(output.root, batch)
                    except Exception as error:
                        batch['errors'].append(dict(query=query, item_id=item_id, stage='listing', error=str(error)))
                        print(f'  Listing failed: {error}', flush=True)
                offset += limit
                if not page.get('next'): break
        batch['state'] = 'complete'
    except KeyboardInterrupt:
        batch['state'] = 'interrupted'
        print('Stopped; completed combined images have been saved.', flush=True)
    finally:
        batch['finished_utc'] = datetime.now(timezone.utc).isoformat()
        checkpoint(output.root, batch)
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-cards', type=int, default=TOTAL_CARDS_ANALYZED)
    parser.add_argument('--max-listings-per-card', type=int, default=MAX_LISTINGS_PER_CARD)
    parser.add_argument('--max-candidates-per-card', type=int, default=MAX_CANDIDATES_PER_CARD)
    parser.add_argument('--listing-age-hours', type=float, default=LISTING_AGE_HOURS)
    parser.add_argument('--local-images', nargs='+', type=Path, help='Offline smoke test; uses separate ebay_line_fit_local folders, no eBay calls.')
    args = parser.parse_args()
    if min(args.total_cards, args.max_listings_per_card, args.max_candidates_per_card) < 1:
        parser.error('Batch limits must be positive integers')
    if args.max_candidates_per_card > 10000:
        parser.error('Max candidates must be <= 10000 (eBay search pagination limit)')
    if args.listing_age_hours is not None and args.listing_age_hours <= 0:
        parser.error('Listing age must be positive, or use None in the configuration')
    # Check imports before any previous batch is cleared.
    from yellow_batch_engine.compare import compare
    if args.local_images:
        paths = [p.resolve(strict=True) for p in args.local_images]
        local_root = OUTPUT_ROOT.with_name('ebay_line_fit_local')
        with BatchOutput(local_root) as output:
            batch = dict(state='complete', records=[], errors=[], source='local smoke test')
            for path in paths[:args.total_cards]:
                print(f'Fitting local image {path.name}', flush=True)
                batch['records'].append(process_front(path.read_bytes(), dict(card_id='local', query=path.name,
                    item_id=str(path), title=path.name, url=''), output.root))
                checkpoint(output.root, batch)
        print(f'Review: {local_root / "review.html"}')
        return
    cards = load_cards()
    if SHUFFLE_CARDS: random.Random(RANDOM_SEED).shuffle(cards)
    client = EbayClient()  # Authenticate successfully before clearing the old images.
    try:
        with BatchOutput(OUTPUT_ROOT) as output:
            batch = run_batch(cards, client, output, total=args.total_cards,
                per_card=args.max_listings_per_card, candidates_per_card=args.max_candidates_per_card,
                age_hours=args.listing_age_hours, exclude_previous=EXCLUDE_PREVIOUS_BATCH)
    finally:
        client.close()
    print(f"\nProcessed {len(batch['records'])}/{args.total_cards} listing fronts.")
    if len(batch['records']) < args.total_cards:
        print('Not enough eligible, downloadable listings within the configured search limits. See batch.json for errors.')
    print(f'Review: {OUTPUT_ROOT / "review.html"}')
    print(f'Image folder: {OUTPUT_ROOT / "Combined"}')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit(f'Batch stopped: {error}') from None
