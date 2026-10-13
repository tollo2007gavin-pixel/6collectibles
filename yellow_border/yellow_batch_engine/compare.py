"""Run both fit families on identical frames and render their measured paths."""
import copy
import textwrap
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from .cardcv.pipeline import analyze, rejection_reasons, DEFAULT_GATES
from .cardcv.orientation import card_orientation
from .cardcv.yellow_curves import attach_boundary_tracks
from .cardcv.photo_margins import boundary_margins
from .exterior_gate import inspect_exterior_background, apply_exterior_gate
from .pale_border import validate_pale_border
from .yellow_validation import validate_yellow_illumination
from .frame_diagnostics import evaluate_frames, combine_frame_diagnostics


def unresolved(reason):
    return {axis: dict(resolved=False, reasons=[reason]) for axis in ('lr', 'tb')}


def fit_seed(image, seed, mode, *, measurement_frame='initial'):
    frame = dict(diag=dict(capture='photo'), result=dict(
        outer=np.asarray(seed['outer']), inner=np.asarray(seed['inner']), px_per_mm=seed['ppm']))
    # Do not grow diagnostic fits around an already rejected physical frame.
    attach_boundary_tracks(image, frame, model_mode=mode,
                           allow_inner_recovery=not seed['geometry_reasons'])
    tracks = {k: frame['diag'][k+'_curve_candidates'] for k in ('outer', 'inner')}
    return measure_seed_tracks(image,seed,tracks,measurement_frame=measurement_frame)


def measure_seed_tracks(image,seed,tracks,*,measurement_frame='initial'):
    """Validate and measure fitted paths without changing their coordinates."""
    if measurement_frame not in ('initial','refit_when_supported'):
        raise ValueError('measurement_frame must be initial or refit_when_supported')
    paths = {k: [t['points'] if t and t['retained'] else None for t in tracks[k]] for k in tracks}
    colour_recovery=validate_pale_border(image,seed,tracks)
    illumination=validate_yellow_illumination(image,seed,tracks) if not colour_recovery['applied'] else None
    if illumination and illumination['applied']:
        colour_recovery=illumination
    geometry_reasons=[] if colour_recovery['applied'] else list(seed['geometry_reasons'])
    axes = {}
    try:
        measured = boundary_margins(seed['outer'], seed['inner'], paths['outer'], paths['inner'], seed['top_side'])
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
        axes = unresolved(str(error))
    else:
        for axis, sides in [('lr', (3, 1)), ('tb', (0, 2))]:
            indices = [(s+seed['top_side']) % 4 for s in sides]
            reasons = list(geometry_reasons)
            missing = [f'{k}:{s}' for k in paths for s in indices if paths[k][s] is None]
            if missing:
                reasons.append('Unresolved boundaries: '+', '.join(missing))
            used = [tracks[k][s] for k in tracks for s in indices if tracks[k][s]]
            key = 'left_pct' if axis == 'lr' else 'top_pct'
            axes[axis] = dict(resolved=not reasons, reasons=reasons,
                percentage=measured['ratios'][key] if not reasons else None,
                support=float(np.mean([t['support'] for t in used])) if used else 0.,
                nominal_margins_mm={k: measured['margins_mm'][k] for k in
                                   (('left', 'right') if axis == 'lr' else ('top', 'bottom'))} if not reasons else None)
    frame_diagnostics=evaluate_frames(seed,tracks)
    for axis in axes:
        diagnostic=frame_diagnostics['axes'][axis]
        diagnostic['active_frame']='initial'
        if measurement_frame=='refit_when_supported' and axes[axis]['resolved'] and diagnostic['candidate_available']:
            axes[axis]['percentage']=diagnostic['candidate_percentage']
            axes[axis]['nominal_margins_mm']=diagnostic['candidate_nominal_margins_mm']
            diagnostic['active_frame']='refit'
        axes[axis]['frame_diagnostics']=diagnostic
    frame_diagnostics['requested_policy']=measurement_frame
    selected_frames={a['frame_diagnostics']['active_frame'] for a in axes.values()}
    frame_diagnostics['active_frame']=next(iter(selected_frames)) if len(selected_frames)==1 else 'mixed'
    return dict(seed=seed, candidates=tracks, paths=paths, axes=axes,colour_recovery=colour_recovery,
                frame_diagnostics=frame_diagnostics,illumination_recovery=illumination)


def combine_trials(trials):
    axes = {}; chosen = {}
    for axis in ('lr', 'tb'):
        good = [(i, t['axes'][axis]) for i, t in enumerate(trials) if t['axes'][axis]['resolved']]
        if not good:
            reasons = list(dict.fromkeys(r for t in trials for r in t['axes'][axis]['reasons']))
            axes[axis] = dict(resolved=False, reasons=reasons or ['No supported frame'])
            if len(trials)==1 and 'frame_diagnostics' in trials[0]['axes'][axis]:
                axes[axis]['frame_diagnostics']=copy.deepcopy(trials[0]['axes'][axis]['frame_diagnostics'])
            continue
        values = np.asarray([v['percentage'] for _, v in good])
        distances = np.abs(values[:, None]-values[None, :]); counts = (distances < 1.5).sum(axis=1)
        if len(good) > 1 and counts.max() == 1:
            axes[axis] = dict(resolved=False, reasons=['Refined candidates disagree by at least 1.5 pp'])
            continue
        ix = max(range(len(good)), key=lambda i: (counts[i], good[i][1]['support'], -float(np.median(distances[i]))))
        trial_index, value = good[ix]; chosen[axis] = trial_index
        axes[axis] = dict(value, selected_trial=trial_index,
                          selected_seed=trials[trial_index]['seed']['method'], candidate_values=values.tolist())
    # The overlay must use the actual source paths that produced each axis.
    display = copy.deepcopy(trials[0]['paths']) if trials else {k: [None]*4 for k in ('outer','inner')}
    for axis, ix in chosen.items():
        t = trials[ix]
        for side in ((3, 1) if axis == 'lr' else (0, 2)):
            s = (side+t['seed']['top_side']) % 4
            for k in display:
                display[k][s] = t['paths'][k][s]
    return dict(axes=axes, paths=display, trials=trials,
                status='Measured' if all(a['resolved'] for a in axes.values()) else 'Review / unresolved')


def compare(image,*,measurement_frame='initial',recover_background=True,recover_boundaries=True):
    """Image is full-resolution, EXIF-corrected BGR. No AI or target ratios."""
    best, candidates = analyze(image, meta=dict(capture='photo', holder='unknown'), fit_tracks=False)
    usable = sorted((c for c in candidates if c.get('ok')), key=lambda c: -c['score'])
    if best is not None:
        selected = [best]
    else:
        selected = [c for c in usable if not rejection_reasons(c, DEFAULT_GATES, meta=dict(holder='unknown'))]
        if not selected:
            selected = usable[:1]  # Still show diagnostic paths, with physical rejection retained.
    seeds = []
    for c in selected:
        r = c['result']; top, _ = card_orientation(image, r['outer'], r['inner'])
        seeds.append(dict(method=c['method'], outer=np.asarray(r['outer']).tolist(),
            inner=np.asarray(r['inner']).tolist(), ppm=r['px_per_mm'], top_side=int(top),
            keystone=c['diag']['keystone'], geometry_reasons=rejection_reasons(c, DEFAULT_GATES, meta=dict(holder='unknown'))))
    modes = {}
    for mode in ('adaptive', 'forced'):
        if not seeds:
            modes[mode] = dict(axes=unresolved('No supported card frame found'), paths={k: [None]*4 for k in ('outer','inner')}, trials=[], status='No fit')
        else:
            modes[mode] = combine_trials([fit_seed(image, seed, mode,measurement_frame=measurement_frame) for seed in seeds])
    analysis = dict(methods=modes, seeds=seeds, candidate_diagnostics=[
        {k: c[k] for k in ('method','ok','stage','reason','reject_reasons','score') if k in c} for c in candidates],
        physical_deformation_corrected=False)
    gate = inspect_exterior_background(image, analysis)
    analysis['image_gate'] = gate
    analysis['methods'] = {m: apply_exterior_gate(r, gate) for m,r in modes.items()}
    if recover_background:
        from .background_recovery import recover_background as recover
        analysis=recover(image,analysis,measurement_frame=measurement_frame)
    if recover_boundaries:
        from .boundary_recovery import recover_boundary_selection
        analysis=recover_boundary_selection(image,analysis,measurement_frame=measurement_frame)
    return analysis


def combine_methods(methods):
    """Prefer forced paths per edge; average only qualified per-axis percentages.

    Adaptive is not necessarily straight. The displayed paths deliberately
    remain observed fits, while the numerical estimate can be a method mean.
    Missing or physically rejected measurements never enter the average.
    """
    paths = {}; path_sources = {}
    for kind in ('outer', 'inner'):
        paths[kind] = []; path_sources[kind] = []
        for side in range(4):
            source = next((m for m in ('forced', 'adaptive')
                           if methods[m]['paths'][kind][side] is not None), None)
            paths[kind].append(copy.deepcopy(methods[source]['paths'][kind][side]) if source else None)
            path_sources[kind].append(source)
    axes = {}
    for axis in ('lr', 'tb'):
        valid = {m: methods[m]['axes'][axis] for m in ('adaptive', 'forced')
                 if methods[m]['axes'][axis]['resolved']}
        if len(valid) == 2:
            values = {m: float(a['percentage']) for m,a in valid.items()}
            axes[axis] = dict(resolved=True, reasons=[], percentage=sum(values.values())/2,
                measurement_source='mean', method_percentages=values,
                disagreement_pp=abs(values['forced']-values['adaptive']))
        elif valid:
            mode, value = next(iter(valid.items()))
            axes[axis] = dict(copy.deepcopy(value), measurement_source=mode,
                              method_percentages={mode: value['percentage']})
        else:
            reasons = list(dict.fromkeys(r for m in methods.values() for r in m['axes'][axis]['reasons']))
            axes[axis] = dict(resolved=False, reasons=reasons, measurement_source='unresolved')
    result = dict(paths=paths, path_sources=path_sources, axes=axes,
                status='Measured' if all(a['resolved'] for a in axes.values()) else 'Review / unresolved')
    for axis in axes:
        diagnostics=combine_frame_diagnostics({m:r['axes'][axis] for m,r in methods.items()})
        if diagnostics is not None:axes[axis]['frame_diagnostics']=diagnostics
    gates = [m['image_gate'] for m in methods.values() if m.get('image_gate')]
    if gates:
        gate = next((g for g in gates if g['fail']), gates[0])
        result = apply_exterior_gate(result, gate)
    return result


def render(image, result, path, *, title, mode):
    """One thin, subpixel-accurate overlay even when the measurement is unresolved."""
    original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    thumb = original.copy(); thumb.thumbnail((1500, 1800), Image.Resampling.LANCZOS)
    canvas = np.asarray(thumb.resize((thumb.width*4, thumb.height*4), Image.Resampling.LANCZOS)).copy()
    scale = np.asarray([thumb.width/original.width, thumb.height/original.height])*4
    for kind, color in [('outer', (0, 217, 78)), ('inner', (255, 35, 55))]:
        for points in result['paths'][kind]:
            if points is None: continue
            # Source centres -> image-edge coordinates -> supersampled pixel centres.
            pts = np.rint(((np.asarray(points)+.5)*scale-.5)*256).astype(np.int32)
            cv2.polylines(canvas, [pts], False, color, 3, cv2.LINE_AA, shift=8)
    fitted = Image.fromarray(canvas).resize(thumb.size, Image.Resampling.LANCZOS)
    try: font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    except OSError: font = ImageFont.load_default(size=19)
    ratios = []
    for axis, label in [('lr', 'L/R'), ('tb', 'T/B')]:
        a = result['axes'][axis]
        ratios.append(f"{label} {a['percentage']:.3f}/{100-a['percentage']:.3f}" if a['resolved'] else f'{label} unresolved')
    width = max(900, fitted.width)
    label = {'adaptive': 'Adaptive', 'forced': 'Forced curve', 'combined': 'Combined'}[mode]
    lines = textwrap.wrap(title, width=max(55, int(width/11)))[:2]
    lines += [f"{label} | {result['status']} | " + '   '.join(ratios), 'Green: exterior | Red: interior | Unresolved edges omitted']
    if mode == 'combined':
        sources = {'mean': 'mean of both methods', 'forced': 'forced curve',
                   'adaptive': 'adaptive', 'unresolved': 'unresolved'}
        lines.append('Margins: ' + ' | '.join(f'{axis.upper()} {sources[a["measurement_source"]]}'
                                              for axis,a in result['axes'].items()))
        lines.append('Lines: forced curve where available; adaptive fallback')
        if any(a.get('frame_diagnostics',{}).get('active_frame') in ('refit','mixed') for a in result['axes'].values()):
            lines.append('Perspective frame: ' + ' | '.join(axis.upper()+' '+a.get('frame_diagnostics',{}).get('active_frame','initial') for axis,a in result['axes'].items()))
    if result.get('image_gate',{}).get('fail'):
        lines.extend(textwrap.wrap(result['image_gate']['reason'],width=max(55,int(width/11))))
    header_height = 16+27*len(lines)
    policy=result.get('measurement_policy')
    if policy:
        lines.append('Measurement: '+policy['category']+' | No centering selection gate')
        allowances=policy['axis_allowance_pp']
        if any(v is not None and v>0 for v in allowances.values()):
            lines.append('Proposed angular allowance: '+' | '.join(axis.upper()+(' unavailable' if value is None else f' +/-{value:g} pp') for axis,value in allowances.items()))
        header_height=16+27*len(lines)
    output = Image.new('RGB', (width, fitted.height+header_height), 'white')
    output.paste(fitted, ((width-fitted.width)//2, header_height))
    ImageDraw.Draw(output).multiline_text((10, 8), '\n'.join(lines), font=font, fill='black', spacing=6)
    output.save(path, quality=97, subsampling=0)
