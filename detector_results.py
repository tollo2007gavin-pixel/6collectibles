"""Build a local visual report from saved predictions and verified corner photos."""
import base64
from html import escape
import json
from pathlib import Path


def render(manifest, output):
    from PIL import Image
    data = json.loads(manifest.read_text(encoding='utf-8'))
    rows = {r['file']:r for r in data['images']}
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    predictions = json.loads((output/'predictions.json').read_text(encoding='utf-8'))
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Detector results</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 1em}'
             'section{display:inline-block;vertical-align:top;width:min(100%,500px);margin:0 1em 1em 0}'
             'svg{width:100%;background:#222}label{margin-right:1em}'
             'body:has(#human:not(:checked)) .human,body:has(#model:not(:checked)) .model{display:none}</style>',
             '<h1>Whitening detector results</h1>',
             '<p>'+('Replay of previously tested cards' if output.name.endswith('-replay') else
                    'Fresh reserved test' if report['mode']=='evaluate' else 'Development validation')+
             f' · Epoch {report["completed_epochs"]} · Confidence cutoff {report["threshold"]:.4g}</p>',
             '<label><input id="human" type="checkbox" checked>Green solid: your chips</label>'
             '<label><input id="model" type="checkbox" checked>Orange dashed: model flags</label><hr>']
    for pred in predictions:
        row = rows[pred['file']]
        path = manifest.parent / pred['file']
        with Image.open(path) as img:
            w,h = img.size
        flags = [b for b,s in zip(pred['boxes'], pred['scores'],strict=True) if s >= report['threshold']]
        name = escape(row['group_id'].removeprefix('group:') + ' · ' + row['corner'])
        uri = base64.b64encode(path.read_bytes()).decode('ascii')
        parts.append(f'<section><h2>{name}</h2><p>{len(row["boxes"])} marked chips · {len(flags)} model flags</p>'
                     f'<svg viewBox="0 0 {w} {h}"><image href="data:image/png;base64,{uri}" width="{w}" height="{h}"/>')
        for boxes,kind,stroke,dash in ((row['boxes'],'human','#00ef73',''),(flags,'model','#ff8c00','stroke-dasharray="3 2"')):
            for x,y,r,b in boxes:
                parts.append(f'<rect class="{kind}" x="{x:g}" y="{y:g}" width="{r-x:g}" height="{b-y:g}" '
                             f'fill="none" stroke="{stroke}" stroke-width="1.5" {dash}/>')
        parts.append('</svg></section>')
    parts.append('</html>')
    page = output/'results.html'
    page.write_text(''.join(parts),encoding='utf-8')
    return page
