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
        run_id=str(inputs.get('tune_run_id','')).strip()
        name=str(inputs.get('frozen_config','')).strip()
        if run_id and not run_id.isdigit():
            raise ValueError('Tune workflow run ID must contain digits only')
        if run_id and name:
            raise ValueError('Enter either the tune run ID or a repository config path, not both')
        if run_id:
            config=local('tune_config')
            return mode,dataset,config
        if not name:
            raise ValueError('Validation requires a successful tune run ID or a frozen config path')
        config=local(name)
        if config.is_dir():
            matches=list(config.rglob('frozen-config.json'))
            if len(matches)!=1:
                raise ValueError(f'Expected one frozen-config.json from the selected tune artifact; found {len(matches)}')
            config=matches[0]
        if not config.is_file():
            raise ValueError('The selected tune artifact does not contain frozen-config.json; choose a successful tune run')
    return mode,dataset,config


def main():
    mode,dataset,config=settings(json.loads(os.environ.get('PIXEL_TRAINING_INPUTS','{}')))
    output=ROOT/'outputs'/'pixel_training'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(ROOT/'pixel_training.py'),mode,'--dataset',str(dataset),'--out',str(output)]
    if config is not None:
        if config.is_dir():
            matches=list(config.rglob('frozen-config.json'))
            if len(matches)!=1:raise ValueError(f'Expected one frozen-config.json from the selected tune artifact; found {len(matches)}')
            config=matches[0]
        command+=['--config',str(config)]
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
