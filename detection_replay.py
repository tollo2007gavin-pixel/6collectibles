"""Offline replay of saved originals and safe GitHub workflow input mapping."""
import json
import os
from pathlib import Path


def candidates(folder):
    from card_review import local_asset
    root = Path(folder).resolve()
    runs = list(root.glob('**/_audit/run.json'))
    if len(runs) != 1:
        raise ValueError('Replay artifact must contain exactly one _audit/run.json')
    run = runs[0]
    out = run.parent.parent
    state = json.loads(run.read_text(encoding='utf-8-sig'))
    for row in state['records']:
        packet = row.get('back_packet', {})
        manifest = local_asset(out, packet.get('manifest'))
        photos = []
        if manifest:
            for p in json.loads(manifest.read_text(encoding='utf-8-sig')).get('photos', []):
                original = local_asset(out, str(manifest.parent / p.get('original_file', '')))
                if original:
                    photos.append(str(original))
        if not photos:
            raise ValueError('Saved originals unavailable for listing ' + row['item_id'])
        meta = {k: row.get(k, '') for k in ('item_id', 'card_id', 'border_type', 'title', 'url', 'query')}
        meta['image_url'] = photos[-1]
        yield meta, {'image': {'imageUrl': photos[-1]}, 'additionalImages': [{'imageUrl': p} for p in photos[:-1]]}


def workflow_inputs(inputs):
    inputs = dict(inputs)
    replay = inputs.pop('replay_run_id', '')
    if replay:
        if inputs.get('pipeline') != 'detection' or not str(replay).isdigit():
            raise ValueError('Replay requires detection mode and a numeric GitHub run ID')
        inputs['replay_folder'] = 'replay_batch'
        inputs['max_back_photos'] = 24
    return inputs


if __name__ == '__main__':
    mapped = workflow_inputs(json.loads(os.environ['CARD_RUN_INPUTS']))
    with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as handle:
        handle.write('CARD_RUN_INPUTS=' + json.dumps(mapped, separators=(',', ':')) + '\n')
