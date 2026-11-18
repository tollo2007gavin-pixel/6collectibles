"""GitHub entry point: compare labeled cached measurements without API calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent


def settings(inputs, root=ROOT):
    mode=inputs.get('mode','baseline')
    if mode not in ('baseline','tune','validate'):
        raise ValueError('Choose baseline, tune, or validate')
    root=Path(root).resolve()
    def local(value):
        path=(root/str(value)).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Choose a data path inside this repository')
        return path
    dataset=local(inputs.get('dataset_path','review_labels'))
    if not dataset.exists():
        raise ValueError('Upload your exported label JSON into review_labels first')
    config=None
    if mode=='validate':
        name=str(inputs.get('frozen_config','')).strip()
        if not name:
            raise ValueError('Validation requires the frozen configuration from a completed development run')
        config=local(name)
        if not config.is_file():
            raise ValueError('Upload the frozen configuration JSON before validation')
    return mode,dataset,config


def main():
    mode,dataset,config=settings(json.loads(os.environ.get('PIXEL_TRAINING_INPUTS','{}')))
    output=ROOT/'outputs'/'pixel_training'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(ROOT/'pixel_training.py'),mode,'--dataset',str(dataset),'--out',str(output)]
    if config is not None:command+=['--config',str(config)]
    subprocess.run(command,cwd=ROOT,check=True)
    message=('## Pixel review learning report\n\n'
             f'Mode: **{mode}**. Download the **pixel-learning-report** artifact for the comparison and configuration.\n\n'
             'Uses saved measurements only: no OpenAI calls, no eBay searches, and no changes to the active detector. '
             'Tuning uses development cards only. Validation is a separate explicit action with frozen settings. '
             'After inspecting validation results, preserve the exposure record and use new reserved cards for the next independent check.\n')
    (output/'WORKFLOW_README.md').write_text(message,encoding='utf-8')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a',encoding='utf-8') as f:f.write(message)
    print(message)


if __name__=='__main__':main()
