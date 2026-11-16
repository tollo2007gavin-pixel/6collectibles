"""Measure connected light patches around extracted cores; no labels or IDs."""
import cv2
import numpy as np


def prepare(crop):
    size=crop.shape[0]
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    smooth=cv2.GaussianBlur(lab,(0,0),max(.5,size/500),borderType=cv2.BORDER_REPLICATE)
    return {str(scale):smooth-cv2.GaussianBlur(lab,(0,0),scale*size/500,
                borderType=cv2.BORDER_REPLICATE) for scale in (8,12)}


def measure(component, responses, edge_details, zone):
    size=component.shape[0];scale=500/size
    ys,xs=np.where(component)
    pad=max(8,round(size*.08))
    x0,x1=max(0,xs.min()-pad),min(size,xs.max()+pad+1)
    y0,y1=max(0,ys.min()-pad),min(size,ys.max()+pad+1)
    core=component[y0:y1,x0:x1]
    # A high-pass core can outline a chip rather than cover its bright center.
    # Allow a tiny normalized interpolation gap, not an arbitrary nearby search.
    distance=cv2.distanceTransform((~core).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    # Four pixels at the standard tile matches the extractor's blue-neighbor
    # radius. This allows a core on a chip's rim to support its adjacent center.
    connection_radius=max(1,size*.008)
    seeds=distance<=connection_radius
    results=[]
    for key,response in responses.items():
        local=response[y0:y1,x0:x1]
        peak=float(local[seeds].max())
        if peak<=0:
            results.append(dict(scale=int(key),status='no_light_patch'));continue
        seed=np.argmax(np.where(seeds,local,-np.inf))
        sy,sx=np.unravel_index(seed,core.shape)
        threshold=max(6.,peak*.5)
        _,labels=cv2.connectedComponents((local>=threshold).astype(np.uint8),8)
        label=labels[sy,sx]
        if label==0:
            results.append(dict(scale=int(key),status='weak_light_patch',peak=peak));continue
        patch=labels==label
        # Re-anchor the contour to the patch's own peak so a peripheral seed
        # cannot flood into the surrounding bright background at a low level.
        peak=float(local[patch].max())
        peak_y,peak_x=np.unravel_index(np.argmax(np.where(patch,local,-np.inf)),patch.shape)
        _,labels=cv2.connectedComponents((local>=max(6.,peak*.5)).astype(np.uint8),8)
        patch=labels==labels[peak_y,peak_x]
        if float(distance[patch].min())>connection_radius:
            results.append(dict(scale=int(key),status='detached_light_patch'));continue
        py,px=np.where(patch);area=int(patch.sum())
        w=int(px.max()-px.min()+1);h=int(py.max()-py.min()+1)
        values=np.linalg.eigvalsh(np.cov(np.stack([px,py]))) if area>2 else np.ones(2)
        clipped=bool((px.min()==0 or px.max()==patch.shape[1]-1 or
                      py.min()==0 or py.max()==patch.shape[0]-1))
        result=dict(scale=int(key),status='clipped' if clipped else 'measured',peak=peak,
            contrast_p75=float(np.percentile(local[patch],75)),
            area_at_500px=area*scale*scale,axis_ratio=float(np.sqrt(max(values[-1],1)/max(values[0],1))),
            fill_fraction=area/(w*h),box_in_corner=[int(x0+px.min()),int(y0+py.min()),w,h],
            core_fraction=float(patch[core].mean()),core_gap=float(distance[patch].min()))
        if edge_details:
            result.update(card_overlap=float(edge_details['possible_interior'][y0:y1,x0:x1][patch].mean()),
                edge_overlap=float(zone[y0:y1,x0:x1][patch].mean()),
                min_outside_distance=float(edge_details['outside_distance'][y0:y1,x0:x1][patch].min()))
        results.append(result)
    return dict(scales=results)
