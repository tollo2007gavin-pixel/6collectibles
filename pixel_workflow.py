"""Dedicated candidate-test workflow. Never runs the whitening API classifier."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from card_review import build_batch

ROOT = Path(__file__).resolve().parent


def settings(inputs):
    shape = inputs.get('geometry', 'triangle')
    if shape not in {'triangle','square','circle','ring'}:
        raise ValueError('Unknown geometry')
    size = float(inputs.get('size_pixels', '12'))
    percentage = float(inputs.get('blue_percentage', '60'))
    count = int(inputs.get('max_listings', 20))
    if not math.isfinite(size) or not 2 <= size <= 100:
        raise ValueError('Size must be 2–100 normalized pixels')
    if not math.isfinite(percentage) or not 0 < percentage <= 100:
        raise ValueError('Blue percentage must be above 0 and at most 100')
    if not 1 <= count <= 500:
        raise ValueError('Listing count must be 1–500')
    replay = str(inputs.get('replay_run_id', '')).strip()
    if replay and not replay.isdigit():
        raise ValueError('Replay run ID must be numeric')
    border = inputs.get('border_type','yellow')
    if border not in {'yellow','gx','all'}:
        raise ValueError('Unknown border design')
    grain = inputs.get('grain_filter',True)
    art=inputs.get('exclude_interior_art',True)
    if type(art) is not bool:
        raise ValueError('Interior art exclusion must be a boolean')
    strategy=inputs.get('candidate_strategy','edge_uncertainty')
    if strategy not in ('edge_uncertainty','legacy'):
        raise ValueError('Unknown candidate strategy')
    if type(grain) is not bool:
        raise ValueError('Grain filter must be a boolean')
    ids = json.loads(inputs.get('card_ids','[]'))
    if not isinstance(ids,list) or any(not isinstance(x,str) for x in ids):
        raise ValueError('Card IDs must be a JSON list of strings')
    # UI gives an actual triangle side length, not its altitude.
    fraction = size/500 * (math.sqrt(3)/2 if shape=='triangle' else 1)
    return dict(geometry=shape,size_pixels_at_500=size,blue_threshold=percentage/100,
                measurement_fraction=fraction,grain_filter=grain,candidate_strategy=strategy,exclude_interior_art=art,
                replay_run_id=replay,max_listings=count,border_type=border,card_ids=ids)


def detection_inputs(config, output_root):
    return dict(pipeline='detection', detection_method='validated_whitening',
                output_mode='testing',prepare_api_only=True,max_whitening_cards=0,
                target_passes=0,output_root=str(output_root),max_listings=config['max_listings'],
                border_type=config['border_type'],card_ids=config['card_ids'])


def main():
    config=settings(json.loads(os.environ.get('PIXEL_TEST_INPUTS','{}')))
    token=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output=ROOT/'outputs'/'pixel_tests'/token
    output.mkdir(parents=True,exist_ok=False)
    if config['replay_run_id']:
        source=ROOT/'replay_batch'
    else:
        source=output/'source_runs'
        env=dict(os.environ)
        env.pop('OPENAI_API_KEY',None)
        env['CARD_RUN_INPUTS']=json.dumps(detection_inputs(config,source))
        subprocess.run([sys.executable,str(ROOT/'run_cards.py')],cwd=ROOT,env=env,check=True)
    runs=list(source.glob('**/_audit/run.json'))
    if len(runs)!=1:
        raise ValueError('Expected exactly one saved run with _audit/run.json')
    run=runs[0]; run_root=run.parent.parent
    # Preserve originals and detection output so this artifact can itself be replayed.
    if config['replay_run_id']:
        destination=output/'source_runs'/run_root.name
        shutil.copytree(run_root,destination)
        run_root=destination
        run=run_root/'_audit'/'run.json'
    state=json.loads(run.read_text(encoding='utf-8-sig'))
    batch=build_batch(run_root.resolve(),state,embed=True)
    batch['cards']=batch['cards'][:config['max_listings']]
    text=json.dumps(batch).replace('<','\\u003c')
    source_html=output/'source.html'
    source_html.write_text('<script id="batch-data" type="application/json">'+text+
        '</script><script id="saved-review" type="application/json">null</script>',encoding='utf-8')
    command=[sys.executable,str(ROOT/'pixel_review.py'),str(source_html),'--out',str(output/'candidate_review'),
             '--surround-threshold',str(config['blue_threshold']),
             '--radius-fraction',str(config['measurement_fraction']),'--geometry',config['geometry']]
    if config['grain_filter']:
        command.append('--grain-filter')
    if config['candidate_strategy']=='edge_uncertainty':
        command.append('--edge-filter')
        if config['exclude_interior_art']:
            command.append('--art-filter')
    subprocess.run(command,cwd=ROOT,check=True)
    (output/'settings.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    results=json.loads((output/'candidate_review'/'results.json').read_text())
    summary={'cards':len(results),'sheets_evaluated':sum(r['status']=='needs_review' for r in results),
             'retained_candidates':sum(r.get('candidate_count',0) for r in results),
             'excluded_candidates':sum(len(r.get('excluded_candidates',[])) for r in results),
             'api_calls':0,'settings':config}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    message=(f"## Pixel candidate test\n\nShape: **{config['geometry']}**; normalized size: "
             f"**{config['size_pixels_at_500']:g} px**; blue threshold: **{config['blue_threshold']:.0%}**.\n\n"
             f"Evaluated {summary['sheets_evaluated']} of {summary['cards']} card sheets. "
             f"Retained {summary['retained_candidates']}; excluded {summary['excluded_candidates']}. "
             "Counts are not accuracy. No whitening API calls.\n\n"
             "Download the card-results-pixel artifact and open candidate_review/review.html "
             "inside its run folder. Red = retained, green = excluded, cyan = sampled shape. "
             "settings.json and results.json record the experiment.\n")
    message+=f"\nCandidate strategy: **{config['candidate_strategy']}**. Choose legacy to restore the pre-edge-filter logic.\n"
    (output/'README.md').write_text(message,encoding='utf-8')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a',encoding='utf-8') as f:f.write(message)
    print(message)

if __name__=='__main__': main()
