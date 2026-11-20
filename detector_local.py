"""Windows launcher for local training, checkpoint continuation, and fresh testing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

ROOT = Path(__file__).resolve().parent
SPACE = ROOT / 'detector_workspace'


def python(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def setup():
    from importlib.metadata import PackageNotFoundError, version
    try:
        if (version('torch').split('+')[0], version('torchvision').split('+')[0]) == ('2.9.1', '0.24.1'):
            return
    except PackageNotFoundError:
        pass
    gpu = shutil.which('nvidia-smi')
    cuda = gpu and subprocess.run([gpu, '-L'], capture_output=True).returncode == 0
    build = 'cu128' if cuda else 'cpu'
    print(f'Installing the local PyTorch {build} build. The GPU download is about 3 GB.', flush=True)
    python('-m', 'pip', 'install', '--no-cache-dir', '--index-url',
           'https://download.pytorch.org/whl/' + build, 'torch==2.9.1', 'torchvision==0.24.1')


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def by_hash(files, expected):
    for file in files:
        with file.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() == expected:
                return file
    raise ValueError('The original dataset or checkpoint is missing. Restore it from your detector_workspace backup.')


def record_tests(runs, exposures):
    for report in runs.glob('*-test/report.json'):
        dest = exposures / (report.parent.name + '.json')
        if dest.exists():
            continue
        data = json.loads(report.read_text(encoding='utf-8'))
        aliases = data['evaluated_aliases']
        if not aliases:
            raise ValueError(f'Missing test identities in {report}')
        save_json(dest, dict(schema='pixel-whitening-validation-exposure-v1', exposed_aliases=aliases))


def dataset(reviews, exposures):
    sources = sorted(reviews.glob('*.html'))
    if not sources:
        raise ValueError(f'Save a marked photo review (.html) in {reviews} first.')
    labels = ROOT / 'review_labels'
    key = hashlib.sha256()
    inputs = [ROOT / 'prepare_detector_data.py', *sources,
              *sorted(labels.glob('*.json*')), *sorted(exposures.glob('*.json'))]
    for path in inputs:
        with path.open('rb') as handle:
            key.update(hashlib.file_digest(handle, 'sha256').digest())
    folder = SPACE / 'datasets' / key.hexdigest()
    if not (folder / 'manifest.json').is_file():
        folder.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder.parent) as temp:
            stage = Path(temp) / 'dataset'
            python('prepare_detector_data.py', '--reviews', *sources, '--labels', labels,
                   '--exposure-dir', exposures, '--out', stage)
            stage.rename(folder)
    return folder / 'manifest.json'


def run(action, epochs):
    reviews, runs, exposures = (SPACE / n for n in ('reviews', 'runs', 'exposures'))
    for folder in (reviews, runs, exposures):
        folder.mkdir(parents=True, exist_ok=True)
    record_tests(runs, exposures)
    checkpoints = sorted(p for p in runs.glob('*-train/checkpoint.pt')
                         if (p.parent / 'report.json').is_file())
    if action != 'new' and not checkpoints:
        raise ValueError('Start with option 1 to create the first checkpoint.')
    setup()
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    if action == 'replay':
        previous = sorted(runs.glob('*-test/report.json'))
        if not previous:
            raise ValueError('Run a fresh reserved test with option 3 before replaying it.')
        saved = json.loads(previous[-1].read_text(encoding='utf-8'))
        manifest = by_hash((SPACE/'datasets').glob('*/manifest.json'), saved['manifest_sha256'])
        model = by_hash(reversed(checkpoints), saved['parent_checkpoint_sha256'])
        out = runs/(stamp + '-replay')
        args = ['evaluate', '--dataset', manifest, '--checkpoint', model, '--out', out]
        if saved.get('previous_checkpoint_sha256'):
            args += ['--compare-checkpoint', by_hash(reversed(checkpoints), saved['previous_checkpoint_sha256'])]
        python('detector_train.py', *args)
        report = out/'REPORT.md'
        report.write_text('Replay of '+previous[-1].parent.name+' on previously tested cards.\n\n'+report.read_text(encoding='utf-8'),encoding='utf-8')
        from detector_results import render
        print('Replayed predictions:', render(manifest, out))
        print('Report:', report)
        return
    manifest = dataset(reviews, exposures)
    out = runs / (stamp + ('-test' if action == 'test' else '-train'))
    args = ['evaluate' if action == 'test' else 'train', '--dataset', manifest, '--out', out]
    if action == 'test':
        items = json.loads(manifest.read_text(encoding='utf-8'))['images']
        if not any(item['split'] == 'test' for item in items):
            raise ValueError('All reserved cards have already been used or none were reviewed. Add a fresh marked photo review before testing.')
    else:
        args += ['--epochs', epochs]
    if action != 'new':
        args += ['--checkpoint', checkpoints[-1]]
        print('Using checkpoint', checkpoints[-1], flush=True)
    if action == 'test':
        latest = json.loads((checkpoints[-1].parent / 'report.json').read_text(encoding='utf-8'))
        expected = latest.get('parent_checkpoint_sha256')
        if expected:
            for candidate in reversed(checkpoints[:-1]):
                with candidate.open('rb') as handle:
                    if hashlib.file_digest(handle, 'sha256').hexdigest() == expected:
                        args += ['--compare-checkpoint', candidate]
                        break
    python('detector_train.py', *args)
    record_tests(runs, exposures)
    from detector_results import render
    print('Review the predictions:', render(manifest, out))
    print('Report:', out / 'REPORT.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['new', 'continue', 'test', 'replay'])
    parser.add_argument('--epochs', type=int, default=3)
    options = parser.parse_args()
    if options.epochs < 1:
        parser.error('Choose at least one epoch.')
    action = options.action
    if action is None:
        reviews = SPACE / 'reviews'
        reviews.mkdir(parents=True, exist_ok=True)
        print('Put marked photo reviews in:', reviews)
        if os.name == 'nt':
            os.startfile(reviews)
        action = {'1':'new', '2':'continue', '3':'test', '4':'replay'}.get(input(
            '1 Train from the beginning\n2 Continue the latest model\n3 Test on fresh reserved cards\n4 Replay the last test with its original model and photos\nChoose 1, 2, 3 or 4: ').strip())
        if action is None:
            parser.exit(1, 'No action selected.\n')
    try:
        SPACE.mkdir(exist_ok=True)
        # A process-owned lock releases automatically if the window closes.
        with (SPACE / 'session.lock').open('a+b') as lock:
            if os.name == 'nt':
                import msvcrt
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    parser.exit(1, 'A detector session is already running.\n')
            os.environ.setdefault('TORCH_HOME', str(SPACE / 'weights'))
            run(action, options.epochs)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Detector session stopped: {exc}\n')
