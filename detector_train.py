"""Train or evaluate a two-class Faster R-CNN on box-reviewed corner photographs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

MODEL = "fasterrcnn_resnet50_fpn_v2-smallanchors-v1"
ANCHORS = (8, 16, 32, 64, 128)
SCORE_GRID = (.01, .0125, .015, .02, .025, .03, .04, .05, .075, .10,
              .20, .30, .40, .50, .60, .70, .80, .90, .95)


def identities(row):
    return {f"group:{row['group_id']}", f"image:{row['image_sha256']}", *row.get("aliases", [])}


def read_manifest(path):
    content = path.read_bytes()
    manifest = json.loads(content)
    if manifest.get("schema") != "whitening-detector-v1":
        raise ValueError("Unsupported detector dataset manifest")
    rows = manifest["images"]
    seen = {}
    for row in rows:
        split = row["split"]
        if not row["group_id"] or split not in ("train", "validation", "test"):
            raise ValueError("Each corner needs a physical card group and split")
        for alias in identities(row):
            if seen.setdefault(alias, split) != split:
                raise ValueError("One physical card or image occurs in different splits")
        if not (path.parent / row['file']).resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("Corner file must be inside the dataset folder")
    if len({r["file"] for r in rows}) != len(rows):
        raise ValueError("Duplicate corner file")
    return rows, hashlib.sha256(content).hexdigest()


def make_model(size, pretrained):
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                               fasterrcnn_resnet50_fpn_v2)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None, min_size=size, max_size=size, box_score_thresh=.01,
        rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=512,
        rpn_pre_nms_top_n_test=600, rpn_post_nms_top_n_test=300)
    model.rpn.anchor_generator = AnchorGenerator(tuple((n,) for n in ANCHORS), ((.5, 1., 2.),)*5)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    # Keep the pretrained residual layers fixed; train pyramid, proposals, and ROI heads.
    for parameter in model.backbone.body.parameters():
        parameter.requires_grad_(False)
    return model


def read_corner(root, row):
    import io
    from PIL import Image
    import torch
    from torchvision.transforms.functional import pil_to_tensor
    raw = (root / row["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
        raise ValueError(f"Corner image checksum mismatch: {row['file']}")
    with Image.open(io.BytesIO(raw)) as image:
        w, h = image.size
        tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.)
    for box in row['boxes'] + row.get('baseline_boxes', []):
        if len(box) != 4 or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError(f"Invalid XYXY box for {row['file']}")
    boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
    return tensor, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


def match_box(a, b):
    # Same location tolerance as the existing pixel report, converted to XYXY.
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter > 0 and (inter / (area_a+area_b-inter) >= .1 or
                          (inter/min(area_a, area_b) >= .5 and max(area_a, area_b)/min(area_a, area_b) <= 25))


def matched_count(preds, truth):
    edges = [[j for j, target in enumerate(truth) if match_box(box, target)] for box in preds]
    owner = {}
    def claim(i, visited):
        for j in edges[i]:
            if j not in visited:
                visited.add(j)
                if j not in owner or claim(owner[j], visited):
                    owner[j] = i
                    return True
        return False
    for i in range(len(edges)):
        claim(i, set())
    return len(owner)


def score(rows, predictions, threshold):
    found = actual = flagged = 0
    groups = {}
    for row, det in zip(rows, predictions, strict=True):
        boxes = [b for b,s in zip(det["boxes"], det["scores"], strict=True) if s >= threshold]
        found += matched_count(boxes, row["boxes"])
        actual += len(row["boxes"])
        flagged += len(boxes)
        groups.setdefault(row['group_id'], []).append((row.get('corner'), bool(boxes), bool(row["boxes"])))
    clean = [v for v in groups.values() if {i[0] for i in v} == {'TL','TR','BL','BR'} and not any(i[2] for i in v)]
    clean_flagged = sum(any(i[1] for i in v) for v in clean)
    return dict(chips_found=found, missed=actual-found, false_alarms=flagged-found,
                precision=found/flagged if flagged else None, recall=found/actual if actual else None,
                f1=2*found/(flagged+actual) if flagged+actual else None,
                fully_reviewed_clean_groups=len(clean), clean_groups_flagged=clean_flagged,
                clean_group_false_alarm_rate=clean_flagged/len(clean) if clean else None)


def predict(model, rows, root, device):
    import torch
    model.eval()
    results = []
    with torch.inference_mode():
        for row in rows:
            image, _ = read_corner(root, row)
            det = model([image.to(device)])[0]
            results.append(dict(file=row["file"], boxes=det["boxes"].cpu().tolist(), scores=det["scores"].cpu().tolist()))
    return results


def baseline_score(rows):
    if not rows or any('baseline_boxes' not in row for row in rows):
        return None
    return score(rows, [dict(boxes=r['baseline_boxes'], scores=[1.]*len(r['baseline_boxes'])) for r in rows], .5)


def train_epoch(model, optimizer, rows, root, device, seed):
    import torch
    order = list(rows)
    random.Random(seed).shuffle(order)
    torch.manual_seed(seed)
    model.train()
    # Batch size is one; preserve the pretrained normalization statistics.
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()
    loss_sum = 0.
    for row in order:
        image, target = read_corner(root, row)
        optimizer.zero_grad(set_to_none=True)
        losses = model([image.to(device)], [{k:v.to(device) for k,v in target.items()}])
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss; keeping last completed checkpoint")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / len(order)


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train','evaluate'])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--compare-checkpoint', type=Path, help='Evaluate the direct parent on the same fresh test corners')
    parser.add_argument('--epochs', type=int, default=1, help='Additional complete epochs')
    parser.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--init', choices=['coco','random'], help='Fresh training only; random is for smoke tests')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.action == "evaluate" and not args.checkpoint: parser.error('Evaluation requires --checkpoint')
    if args.compare_checkpoint and args.action != 'evaluate': parser.error('Paired parent comparison is available only in evaluate')
    if args.epochs < 1 or args.threads < 1: parser.error('Epochs and threads must be positive')
    if args.image_size is not None and not 96 <= args.image_size <= 1536: parser.error('Image size must be 96–1536')
    if args.checkpoint and args.init: parser.error('A checkpoint already specifies initialization')
    return args


def main(argv=None):
    args = parse(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torchvision
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows, dataset_sha = read_manifest(args.dataset)
    prior = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else {}
    if args.checkpoint and prior.get('model') != MODEL: raise ValueError('Incompatible detector checkpoint')
    if args.compare_checkpoint:
        with args.compare_checkpoint.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest() != prior.get('parent_checkpoint_sha256'):
                raise ValueError('Comparison requires the exact parent checkpoint used to continue training')
    seed = prior.get('seed', 2026) if args.seed is None else args.seed
    size = prior.get('image_size', 512) if args.image_size is None else args.image_size
    if prior and (seed != prior['seed'] or size != prior['image_size']):
        raise ValueError("A resumed run must use its checkpoint's seed and image size")
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable; use --device cpu')
    train = [r for r in rows if r['split']=='train']
    val = [r for r in rows if r['split']=='validation']
    test = [r for r in rows if r['split']=='test']
    fit_ids = set(prior.get('fit_aliases', []))
    tuning_ids = set(prior.get('tuning_aliases', []))
    def keys(items): return {k for row in items for k in identities(row)}
    if keys(test) & (fit_ids | tuning_ids):
        raise ValueError('Test set overlaps the development history of this checkpoint')
    if keys(val) & fit_ids:
        raise ValueError('Validation set overlaps earlier training groups')
    if args.action=='train' and not any(r['boxes'] for r in train):
        raise ValueError('Training requires a reviewed training corner with whitening')
    if args.action=='evaluate' and not test: raise ValueError('No fresh test corners in this dataset')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    init = prior.get('init',args.init or 'coco')
    model = make_model(size, not prior and init=='coco')
    if prior: model.load_state_dict(prior.pop('weights'))
    model.to(device)
    epoch = prior.get('epoch', 0)
    threshold = prior.get('threshold', .5)
    report = dict(mode=args.action,model=MODEL,manifest_sha256=dataset_sha,
                  device=device,torch=str(torch.__version__),torchvision=str(torchvision.__version__),
                  init=init,seed=seed,image_size=size,anchors=ANCHORS,starting_epoch=epoch)
    if args.checkpoint:
        with args.checkpoint.open('rb') as f: report['parent_checkpoint_sha256'] = hashlib.file_digest(f,'sha256').hexdigest()
    if args.action == 'train':
        opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.001, momentum=.9, weight_decay=.0005)
        if prior: opt.load_state_dict(prior.pop('optimizer'))
        fit_ids.update(keys(train))
        tuning_ids.update(keys(val))
        report.update(train_corners=len(train),train_groups=len({r['group_id'] for r in train}),losses=[])
        def save():
            record = dict(model=MODEL,weights=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                          threshold=threshold,seed=seed,image_size=size,init=init,manifest_sha256=dataset_sha,
                          parent_checkpoint_sha256=report.get('parent_checkpoint_sha256'),
                          fit_aliases=sorted(fit_ids),tuning_aliases=sorted(tuning_ids))
            temp = args.out/'checkpoint.tmp'
            torch.save(record,temp)
            temp.replace(args.out/'checkpoint.pt')
        for _ in range(args.epochs):
            epoch += 1
            loss = train_epoch(model,opt,train,args.dataset.parent,device,seed+epoch)
            report['losses'].append(loss)
            print(f'Epoch {epoch}: mean training loss {loss:.4f}', flush=True)
            save()
        scoring = val
    else:
        scoring = test
        report['evaluated_aliases'] = sorted(keys(test))
    predictions = predict(model, scoring, args.dataset.parent, device)
    if args.action=='train' and any(row['boxes'] for row in val):
        def rank(t):
            s = score(val, predictions,t)
            return s['f1'] or 0, s['recall'] or 0, -abs(t-.5)
        threshold = max(SCORE_GRID, key=rank)
        save()
    report.update(completed_epochs=epoch,threshold=threshold,scored_corners=len(scoring),
                  scored_groups=len({r['group_id'] for r in scoring}),
                  detector=score(scoring,predictions,threshold) if scoring else None,
                  pixel_baseline=baseline_score(scoring))
    if args.compare_checkpoint:
        parent = torch.load(args.compare_checkpoint, map_location='cpu', weights_only=True)
        if parent['model'] != MODEL or parent['image_size'] != size:
            raise ValueError('Incompatible parent detector')
        if keys(test) & (set(parent['fit_aliases']) | set(parent['tuning_aliases'])):
            raise ValueError('Test groups overlap the parent checkpoint development history')
        model.load_state_dict(parent.pop('weights'))
        parent_predictions = predict(model, scoring, args.dataset.parent, device)
        report.update(previous_detector=score(scoring, parent_predictions, parent['threshold']),
                      previous_epoch=parent['epoch'], previous_threshold=parent['threshold'],
                      previous_checkpoint_sha256=prior['parent_checkpoint_sha256'])
        (args.out/'previous-predictions.json').write_text(json.dumps(parent_predictions),encoding='utf-8')
    (args.out/'predictions.json').write_text(json.dumps(predictions),encoding='utf-8')
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    label = 'Development validation (used for confidence selection)' if args.action=='train' else 'Frozen test'
    lines=['# Whitening detector', '', f'{label}. Epoch {epoch}, confidence cutoff {threshold:.4g}.',
           f'Scored {len(scoring)} corners from {report["scored_groups"]} groups.', '',
           '| Model | Found | Missed | False alarms | Precision | Recall | Clean cards flagged |',
           '|---|---:|---:|---:|---:|---:|---:|']
    detectors = [('Faster R-CNN',report['detector']),('Pixel baseline',report['pixel_baseline'])]
    if args.compare_checkpoint:
        detectors.insert(1, (f"Previous Faster R-CNN (epoch {report['previous_epoch']}, cutoff {report['previous_threshold']:.4g})",
                             report['previous_detector']))
    for name,metrics in detectors:
        if metrics is not None:
            fmt=lambda v:'—' if v is None else f'{v:.1%}'
            clean = f"{metrics['clean_groups_flagged']}/{metrics['fully_reviewed_clean_groups']}" if metrics['fully_reviewed_clean_groups'] else '—'
            lines.append(f"| {name} | {metrics['chips_found']} | {metrics['missed']} | {metrics['false_alarms']} | {fmt(metrics['precision'])} | {fmt(metrics['recall'])} | {clean} |")
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__ == '__main__':
    try: main()
    except (ValueError,OSError) as exc: sys.exit(str(exc))
