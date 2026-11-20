"""Add portable spatial human labels to an existing pixel inspection review."""
import hashlib
import html
import json
from pathlib import Path


def add_annotations(page, results, batch):
    if 'id="pixel-annotation-state"' in page:
        return page
    available={c['number'] for c in batch['cards']
               if any(i.get('label')=='Corner close-ups' for i in c.get('images',[]))}
    cards=[]
    for r in results:
        if r['number'] not in available or not r.get('source_sha256'):
            continue
        note=r.get('human_review',{}).get('notes','')
        cards.append(dict(number=r['number'],listing=r['listing'],source_sha256=r['source_sha256'],
            group_id=r['listing'],split='development',notes=note,complete_corners=[],annotations=[],
            previously_reviewed=bool(note),result=r))
        number=r['number']
        editor=f'''<details class="pixel-annotation-editor" id="pixel-editor-{number}">
<summary>Draw whitening boxes</summary>
<div class="pixel-annotation-tools">
<label>Corner <select data-tool="corner"><option>TL</option><option>TR</option><option>BL</option><option>BR</option></select></label>
<label>Mark as <select data-tool="label"><option value="whitening">Whitening</option><option value="uncertain">Uncertain — leave out of scoring</option></select></label>
<label>Zoom <select data-tool="zoom"><option value="1">1×</option><option value="1.5">1.5×</option><option value="2">2×</option><option value="3">3×</option></select></label>
<label><input data-tool="show-candidates" type="checkbox" checked> Show candidate boxes</label>
<button type="button" data-tool="undo">Undo last edit</button>
<button type="button" data-tool="delete">Delete selected mark</button>
</div>
<p>Check all four corners. Click a blue box on a real chip or drag a green box around a missed chip. Use one box per chip. Leave glare and dust alone. If the whole card has no chips, write "No whitening" in its notes. Any mark or note includes the card in your review. On included cards, untouched candidates count as false alarms if the detector flags them.</p>
<div class="pixel-annotation-scroll"><canvas data-tool="canvas" aria-label="Draw labels on the selected card corner"></canvas></div>
<details><summary>Card grouping options</summary>
<div class="pixel-annotation-tools">
<label>Physical card group <input data-tool="group" value="{html.escape(r['listing'],quote=True)}" aria-label="Same physical card group"></label>
<label>Use these labels for <select data-tool="split"><option value="development">Development / tuning</option><option value="validation">Reserved validation</option></select></label>
</div>
<p>Use the same group for every photo of the same physical card. Reserve unfamiliar cards before marking them. Previously discussed cards belong in development. Group and split lock once labeling begins.</p></details>
<p data-tool="status" role="status"></p><ol data-tool="marks"></ol>
</details>'''
        heading=f'<h2>Card {number}</h2>'
        page=page.replace(heading,heading+editor,1)
    controls='''<div id="pixel-annotation-controls">
<strong>Review the batch</strong>
<p>Open <b>Draw whitening boxes</b> on each card. Fresh card groups are automatically split for learning and checking. Review both groups the same way.</p>
<button type="button" id="pixel-label-export">Download boxes for GitHub</button>
<button type="button" id="pixel-label-save">Save marked review with photos</button>
<p><a href="https://github.com/tollo2007gavin-pixel/6collectibles/upload/main/review_labels" target="_blank" rel="noopener noreferrer">Upload the boxes file to GitHub</a> and commit it to <b>main</b> to start the comparison. Keep the photo review if you want to resume editing.</p>
<details><summary>Review options</summary>
<button type="button" id="pixel-label-reserve">Reserve about 30% of unreviewed card groups</button>
<button type="button" id="pixel-label-retire">Move reserved cards to development</button>
<label>Import labels <input type="file" id="pixel-label-import" accept=".json,.gz,application/json,application/gzip"></label>
</details>
<p id="pixel-label-status" role="status"></p>
</div>'''
    # Keep the earlier notes-only download accessible without competing with the boxes export.
    marker='<div id="pixel-review-notes-controls"'
    if marker in page:
        start=page.index(marker)
        end=page.index('</div>',start)+len('</div>')
        page=page[:start]+controls+'<details><summary>Notes-only export</summary>'+page[start:end]+'</details>'+page[end:]
    if 'id="pixel-annotation-controls"' not in page:
        page=controls+page
    identities=sorted((str(c['listing']), c['source_sha256']) for c in cards)
    batch_key=hashlib.sha256(json.dumps(identities).encode('utf-8')).hexdigest()[:16]
    data=json.dumps(dict(schema='pixel-whitening-labels-v1',cards=cards,auto_reserve=True,batch_key=batch_key)).replace('<','\\u003c')
    script=Path(__file__).with_name('pixel_annotations.js').read_text(encoding='utf-8')
    style='''<style>
.pixel-annotation-editor{margin:12px 0;padding:12px;border:1px solid #aaa;background:#fff}
.pixel-annotation-tools{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:12px 0}
.pixel-annotation-tools label{display:flex;gap:5px;align-items:center}
.pixel-annotation-scroll{overflow:auto;max-height:75vh;background:#ddd;border:1px solid #aaa}
.pixel-annotation-scroll canvas{display:block;touch-action:none;max-width:none;max-height:none;cursor:crosshair}
#pixel-annotation-controls{padding:12px;background:#e9f3fa;border:1px solid #779;margin:12px 0}
.pixel-annotation-editor li button{margin-left:8px}
</style>'''
    return page+style+'<script id="pixel-annotation-state" type="application/json">'+data+'</script><script>'+script+'</script>'
