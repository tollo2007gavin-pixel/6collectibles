"""Independent input perturbations; repeatability evidence, not accuracy gates."""
import cv2
import numpy as np
from .engine import analyze


def inspect(image, baseline, *, holder='unknown',capture='photo'):
    transforms=[('resize75',cv2.resize(image,None,fx=.75,fy=.75)),
                ('jpeg85',cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])[1],cv2.IMREAD_COLOR))]
    rows=[]
    for name,modified in transforms:
        result=analyze(modified,holder=holder,capture=capture,shadow=False)
        row=dict(transform=name,status=result['status'],axes=result['axes'],axis_change_pp={},source_corner_shift_px={},reasons=result['reasons'])
        for axis in ['lr','tb']:
            before=baseline.get('axes',{}).get(axis,{});after=result['axes'].get(axis,{})
            if before.get('measured') and after.get('measured'):row['axis_change_pp'][axis]=after['first_pct']-before['first_pct']
        a,b=baseline.get('geometry'),result.get('geometry')
        if a and b:
            # Use actual decoded dimensions; resize rounds pixel dimensions.
            scale=np.array([image.shape[1]/modified.shape[1],image.shape[0]/modified.shape[0]])
            for kind in ['outer','inner']:
                row['source_corner_shift_px'][kind]=np.linalg.norm(np.array(b[kind])*scale-np.array(a[kind]),axis=1).tolist()
        rows.append(row)
    return dict(variants=rows,status_changed=any(r['status']!=baseline['status'] for r in rows),
        note='Report only; neither stable fits nor numerical agreement establishes correct boundary identity. No calibrated GX tolerance is asserted.')
