"""Portable, local-only per-card notes for pixel inspection reports."""
import html
import json
import re


def add_review_notes(page, results):
    if 'id="pixel-review-notes-controls"' in page:
        return page
    records=[]
    for r in results:
        number=r['number']
        records.append({k:r[k] for k in ('number','listing','source_sha256','status','whitening_classification') if k in r})
        note=r.get('human_review',{}).get('notes','')
        field=(f'<div class="pixel-review-note"><label for="pixel-note-{number}"><strong>Your review notes</strong></label>'
               f'<textarea id="pixel-note-{number}" data-card-number="{number}" rows="4" '
               'placeholder="Mention candidate numbers, missed whitening, glare, or other observations." '
               f'style="display:block;box-sizing:border-box;width:100%;margin:8px 0 20px;font:inherit">{html.escape(str(note))}</textarea></div>')
        marker=f'<h2>Card {number}</h2>'
        page=page.replace(marker,marker+field,1)
    controls='''<div id="pixel-review-notes-controls" style="position:sticky;top:0;background:#fff;padding:12px;border:1px solid #bbb;z-index:10">
<button type="button" id="pixel-save-html">Save review with photos and notes</button>
<button type="button" id="pixel-save-json">Export notes JSON</button>
<span id="pixel-notes-status" role="status"></span>
<p style="margin:6px 0 0">Notes stay on this page. Save before closing; the downloaded HTML keeps your photos and editable notes together.</p></div>'''
    page=re.sub(r'(<h1>.*?</h1>)',lambda m:m[0]+controls,page,count=1)
    data=json.dumps(records).replace('<','\\u003c')
    return page+'<script id="pixel-review-records" type="application/json">'+data+'''</script>
<script>
(()=>{
const boxes=()=>Array.from(document.querySelectorAll('.pixel-review-note textarea'));
const status=document.getElementById('pixel-notes-status');
function download(content,type,name){
 const url=URL.createObjectURL(new Blob([content],{type}));
 const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),10000);
 status.textContent='Download requested. Keep the downloaded file to preserve your notes.';
}
boxes().forEach(box=>box.addEventListener('input',()=>{status.textContent='Unsaved changes — save your review.';}));
document.getElementById('pixel-save-html').addEventListener('click',()=>{
 const copy=document.documentElement.cloneNode(true);
 boxes().forEach(box=>{copy.querySelector('#'+box.id).textContent=box.value;});
 copy.querySelector('#pixel-notes-status').textContent='Saved review loaded. Save again after editing.';
 download('<!doctype html>\\n'+copy.outerHTML,'text/html;charset=utf-8','reviewed-pixel-inspection.html');
});
document.getElementById('pixel-save-json').addEventListener('click',()=>{
 const cards=JSON.parse(document.getElementById('pixel-review-records').textContent);
 cards.forEach(card=>{card.human_review={notes:document.getElementById('pixel-note-'+card.number).value};});
 download(JSON.stringify({schema:'pixel-review-notes-v1',reviewed_at:new Date().toISOString(),cards},null,2),
 'application/json','pixel-review-notes.json');
});
})();
</script>'''
