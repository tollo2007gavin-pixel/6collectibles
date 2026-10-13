"""Photo border widths from observed paths, with a separate straight cross-check."""
import cv2
import numpy as np
from .geom import CARD_W_MM,CARD_H_MM


def boundary_margins(outer,inner,outer_paths,inner_paths,top_side,*,trim=.10,axes=('lr','tb')):
    """Mean directional widths over a common central span, in card orientation.

    Retains the outer-quad homography; this is not surface/lens dewarping.
    Paths replace only supported sides. No extrapolation past a path's ends.
    """
    quads={'outer':np.asarray(outer,float),'inner':np.asarray(inner,float)}
    target=np.float32([[0,0],[CARD_W_MM,0],[CARD_W_MM,CARD_H_MM],[0,CARD_H_MM]])
    matrix=cv2.getPerspectiveTransform(np.roll(quads['outer'],-top_side,axis=0).astype(np.float32),target)
    source_paths={'outer':outer_paths or [None]*4,'inner':inner_paths or [None]*4}
    rect={};coverage={}
    for kind in quads:
        rect[kind]=[];coverage[kind]=[]
        for side in range(4):
            i=(side+top_side)%4;path=source_paths[kind][i]
            points=np.asarray(path if path is not None else [quads[kind][i],quads[kind][(i+1)%4]],float)
            homogeneous=np.column_stack([points,np.ones(len(points))])@matrix.T
            if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:,2])<1e-9):
                raise ValueError('invalid projected boundary')
            rect[kind].append(homogeneous[:,:2]/homogeneous[:,2:])
            coverage[kind].append(path is not None)
    margins={};details={};names=('top','right','bottom','left')
    for axis,sides,along,length in [('lr',(3,1),1,CARD_H_MM),('tb',(0,2),0,CARD_W_MM)]:
        if axis not in axes:continue
        bounds=[]
        for kind in rect:
            for side in sides:
                values=rect[kind][side][:,along];difference=np.diff(values)
                if not (np.all(difference>0) or np.all(difference<0)):
                    raise ValueError('boundary is not monotonic along the measurement axis')
                bounds.append((values.min(),values.max()))
        low=max(length*trim,max(a for a,b in bounds));high=min(length*(1-trim),min(b for a,b in bounds))
        if high-low<.6*length-1e-6:raise ValueError('insufficient common observed boundary span')
        positions=np.linspace(low,high,401);coordinates={}
        for kind in rect:
            for side in sides:
                p=rect[kind][side];order=np.argsort(p[:,along])
                coordinates[kind,side]=np.interp(positions,p[order,along],p[order,1-along])
        a,b=sides
        widths=[coordinates['inner',a]-coordinates['outer',a],coordinates['outer',b]-coordinates['inner',b]]
        if not all(np.isfinite(w).all() and np.min(w)>0 for w in widths):
            raise ValueError('inner and outer boundaries cross')
        for side,width in zip(sides,widths):
            margins[names[side]]=float(np.trapezoid(width,positions)/(high-low))
        local_ratio=100*widths[0]/(widths[0]+widths[1])
        details[axis]={'all_boundaries_tracked':all(coverage[k][s] for k in coverage for s in sides),
                       'sampled_fraction':[float(low/length),float(high/length)],
                       'local_ratio_p10_p90':np.percentile(local_ratio,[10,90]).tolist()}
    ratios={}
    for a,b in [('left','right'),('top','bottom')]:
        if a not in margins:continue
        ratios[a+'_pct']=100*margins[a]/(margins[a]+margins[b])
        ratios[b+'_pct']=100-ratios[a+'_pct']
    return {'margins_mm':margins,'ratios':ratios,'axes':details}


def apply_photo_measurement(result):
    """Report track-sampled photo margins and retain the independent old values."""
    result['measurement_method']='straight_frame'
    if result.get('capture')!='photo' or not any(result.get(k+'_boundary_paths') for k in ('outer','inner')):
        return result
    try:
        measured=boundary_margins(result['outer'],result['inner'],result.get('outer_boundary_paths'),
                                 result.get('inner_boundary_paths'),result['top_side'])
    except (ValueError,cv2.error,np.linalg.LinAlgError) as error:
        result['quality']['review_required']=True
        result['quality'].setdefault('review_reasons',[]).append('Photo margin sampling failed: '+str(error))
        result['quality']['measurement_geometry']='straight_frame_requires_review'
        return result
    result['straight_measurement']={'centering':dict(result['centering']),'margins_mm':dict(result['margins_mm'])}
    quality=result['quality'];measured['straight_disagreement_pp']={k:measured['ratios'][k]-result[k] for k in ('left_pct','top_pct')}
    half_paths={kind:[t.get('half_resolution_points') if t else None for t in quality.get(kind+'_curve_tracks',[None]*4)] for kind in ('outer','inner')}
    try:
        half=boundary_margins(result['outer'],result['inner'],half_paths['outer'],half_paths['inner'],result['top_side'])
        measured['resampling_shift_pp']={k:float(half['ratios'][k]-measured['ratios'][k]) for k in ('left_pct','top_pct')}
    except (ValueError,cv2.error,np.linalg.LinAlgError):measured['resampling_shift_pp']=None
    for axis,detail in measured['axes'].items():
        if not detail['all_boundaries_tracked']:
            quality.setdefault('review_reasons',[]).append(axis.upper()+' includes an unresolved side using its straight fallback')
    measured['physical_deformation_corrected']=False
    result['photo_measurement']=measured
    result['margins_mm']=measured['margins_mm'];result.update(measured['ratios'])
    result['centering']=dict(result['centering'],**measured['ratios'],margins_in={k:v/25.4 for k,v in measured['margins_mm'].items()})
    result['measurement_method']='photo_boundary_tracks'
    quality['measurement_geometry']='observed_boundary_tracks'
    return result
