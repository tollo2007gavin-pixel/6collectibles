"""Conservative background rejection from several blue-card silhouettes.

An agreement region is evidence, not a calibrated probability. No original pixels
are cropped. Invalid/disagreeing estimates fail open for human review.
"""
import cv2
import numpy as np


def estimate(crop, corner, margin_fraction=.04, details=None):
    size=crop.shape[0]
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    plausible=(hsv[:,:,0]>=85)&(hsv[:,:,0]<=140)&(hsv[:,:,2]>20)
    sats=hsv[:,:,1][plausible]
    empty=np.zeros(crop.shape[:2],bool)
    def failed(reason):
        return dict(status='unconfirmed',reason=reason,margin_pixels=size*margin_fraction),empty,empty
    if sats.size<size*size*.08:
        return failed('insufficient_blue')
    p90=float(np.percentile(sats,90))
    masks=[]
    inward={'TL':(size-1,size-1),'TR':(0,size-1),'BL':(size-1,0),'BR':(0,0)}[corner]
    k=max(3,round(size*.014)|1)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    for factor in (.55,.65,.75):
        blue=(plausible & (hsv[:,:,1]>max(65,p90*factor))).astype(np.uint8)
        blue=cv2.morphologyEx(blue,cv2.MORPH_OPEN,kernel)
        contours,_=cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:return failed('missing_silhouette')
        contour=max(contours,key=cv2.contourArea)
        if cv2.contourArea(contour)<size*size*.12:return failed('small_silhouette')
        # Blue needs to reach both inward image sides; a detached blue object
        # must not become a fabricated card just by adding the inward corner.
        x,y,w,h=cv2.boundingRect(contour)
        inward_x = x if inward[0]==0 else size-(x+w)
        inward_y = y if inward[1]==0 else size-(y+h)
        if max(inward_x,inward_y)>size*.12:return failed('not_anchored_to_interior')
        points=np.concatenate([contour,np.array([[inward]],np.int32)])
        mask=np.zeros(crop.shape[:2],np.uint8)
        cv2.fillConvexPoly(mask,cv2.convexHull(points),1)
        masks.append(mask.astype(bool))
    union=np.logical_or.reduce(masks)
    intersection=np.logical_and.reduce(masks)
    disagreement=float(np.count_nonzero(union ^ intersection)/max(1,union.sum()))
    if disagreement>.25:return failed('silhouettes_disagree')
    # Margin protects mild protrusions and interpolation/outline errors. The
    # differing silhouettes further widen the zone where they disagree locally.
    margin=max(4,round(size*margin_fraction))
    outside=cv2.distanceTransform((~union).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    padded=cv2.copyMakeBorder(intersection.astype(np.uint8),margin+2,margin+2,margin+2,margin+2,cv2.BORDER_REPLICATE)
    inside=cv2.distanceTransform(padded,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[margin+2:-margin-2,margin+2:-margin-2]
    background=outside>margin
    zone=(outside<=margin)&(inside<=margin)
    # Bright neutral regions touching the edge may be washed-out card/glare.
    # Do not use missing blue alone to suppress them. White backgrounds can
    # consequently disable rejection, which is preferable to hiding damage.
    bright=((hsv[:,:,2]>=150)&(hsv[:,:,1]<100)).astype(np.uint8)
    bright=cv2.dilate(bright,np.ones((3,3),np.uint8))
    _,labels=cv2.connectedComponents(bright,8)
    touching=np.unique(labels[zone & (bright>0)])
    touching=touching[touching>0]
    glare=np.isin(labels,touching) if touching.size else np.zeros_like(zone)
    background &= ~glare
    zone |= glare
    info=dict(status='agreement',silhouettes=3,disagreement_fraction=disagreement,
              margin_pixels=margin,background_fraction=float(background.mean()),
              uncertainty_fraction=float(zone.mean()))
    info['bright_connected_fraction']=float(glare.mean())
    if details is not None:
        details.update(possible_interior=union,unanimous_interior=intersection,
                       outside_distance=outside,glare_uncertainty=glare)
    return info,background,zone


def classify(component, background, zone):
    # An entire connected spot must be clear of the uncertainty zone. A white
    # protrusion attached to the card is retained even if most pixels lie outside.
    return dict(background_fraction=float(background[component].mean()),
                uncertainty_fraction=float(zone[component].mean()),
                exclude=bool(np.all(background[component])))
