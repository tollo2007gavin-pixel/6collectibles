"""Experimental bright-spot measurements on existing four-corner tip sheets."""
import cv2
import numpy as np


def inspection_band(name, crop):
    """Measure the blue silhouette in each saved crop and offset it proportionally.

    Opening removes isolated blue noise; the hull bridges small missing-ink chips.
    A failed segmentation yields empty masks rather than a fabricated boundary.
    """
    size = crop.shape[0]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue_blue = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 2] > 20)
    # Relative saturation separates blue ink from a gray sleeve tinted blue.
    sats = hsv[:, :, 1][hue_blue]
    sat_floor = max(65, float(np.percentile(sats, 90))*.65) if sats.size else 65
    blue = (hue_blue & (hsv[:, :, 1] > sat_floor)).astype(np.uint8)
    radius = max(3, int(round(size*.014)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    interior = np.zeros((size, size), np.uint8)
    if not contours:
        return interior.astype(bool), interior.astype(bool)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < size*size*.08:
        return interior.astype(bool), interior.astype(bool)
    # The inward crop corner is card interior, including non-blue printed artwork.
    inward = {'TL': (size-1, size-1), 'TR': (0, size-1),
              'BL': (size-1, 0), 'BR': (0, 0)}[name]
    points = np.concatenate([contour, np.array([[inward]], dtype=np.int32)])
    cv2.fillConvexPoly(interior, cv2.convexHull(points), 1)
    # Replicate crop borders so image cut lines are not mistaken for card edges.
    padded = cv2.copyMakeBorder(interior, 30, 30, 30, 30, cv2.BORDER_REPLICATE)
    inside = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[30:-30, 30:-30]
    # Only an inner cutoff: all pixels outward of it remain eligible.
    band = inside <= size*.024
    return band, interior.astype(bool)


def corner_tiles(sheet):
    """Read card_utils.corner_closeup_sheet crops exactly, excluding labels/gaps."""
    if sheet is None or sheet.ndim != 3 or sheet.shape[2] != 3:
        raise ValueError('Expected a readable color corner sheet')
    height, width = sheet.shape[:2]
    tile = (width-10)//2
    if tile < 32 or width != tile*2+10 or height != tile*2+50:
        raise ValueError('Unsupported layout; expected the saved four-corner tip sheet')
    return [(name, x, y, sheet[y:y+tile, x:x+tile]) for name, x, y in
            [('TL', 0, 20), ('TR', tile+10, 20),
             ('BL', 0, tile+50), ('BR', tile+10, tile+50)]]


def blue_surround(component, hsv, radius_fraction=.02):
    """Measure a perimeter ring; incomplete rings cannot establish enclosure."""
    ys, xs = np.where(component)
    radius = max(2, round(component.shape[0]*radius_fraction))
    x0, x1 = max(0, xs.min()-radius-1), min(component.shape[1], xs.max()+radius+2)
    y0, y1 = max(0, ys.min()-radius-1), min(component.shape[0], ys.max()+radius+2)
    local = component[y0:y1, x0:x1]
    distance = cv2.distanceTransform((~local).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ring = (distance > 0) & (distance <= radius)
    color = hsv[y0:y1, x0:x1]
    # Dark blue remains blue; neutral white/gray consumes the allowed non-blue fraction.
    plausible = (hsv[:,:,0] >= 85) & (hsv[:,:,0] <= 140) & (hsv[:,:,2] >= 20)
    sats = hsv[:,:,1][plausible]
    sat_floor = max(65, float(np.percentile(sats,90))*.65) if sats.size else 65
    blue = (color[:,:,0] >= 85) & (color[:,:,0] <= 140) & (color[:,:,1] >= sat_floor) & (color[:,:,2] >= 20)
    yy, xx = np.indices(local.shape)
    angles = (np.arctan2(yy-(ys.mean()-y0), xx-(xs.mean()-x0))+2*np.pi) % (2*np.pi)
    sectors = np.minimum((angles/(np.pi/4)).astype(int), 7)
    fractions = [float(blue[ring & (sectors==i)].mean()) if np.any(ring & (sectors==i)) else 0.0 for i in range(8)]
    complete = bool(xs.min() >= radius and ys.min() >= radius and
                    xs.max()+radius < component.shape[1] and ys.max()+radius < component.shape[0])
    return dict(radius_pixels=radius, blue_fraction=float(blue[ring].mean()) if ring.any() else 0.0,
                sector_blue_fractions=fractions, minimum_sector_blue_fraction=min(fractions), complete_ring=complete)


def triangle_blue(component, hsv, corner, altitude_fraction=.02, shape='triangle'):
    """Equilateral triangle with apex at centroid and base toward the named corner."""
    ys, xs = np.where(component)
    apex = np.array([xs.mean(), ys.mean()])
    altitude = component.shape[0]*altitude_fraction
    theta = {'TR':45, 'TL':135, 'BL':225, 'BR':315}[corner]
    radians = np.deg2rad(theta)
    direction = np.array([np.cos(radians), -np.sin(radians)])
    perpendicular = np.array([-direction[1], direction[0]])
    base = apex + altitude*direction
    vertices = np.array([apex, base+altitude/np.sqrt(3)*perpendicular,
                         base-altitude/np.sqrt(3)*perpendicular])
    if shape in ('square', 'circle'):
        half = altitude/2
        vertices = apex + np.array([[-half,-half],[half,-half],[half,half],[-half,half]])
    complete = bool(np.all(vertices >= 0) and np.all(vertices[:,0] <= component.shape[1]-1)
                    and np.all(vertices[:,1] <= component.shape[0]-1))
    # Test pixel centers against all three half-planes, avoiding rounded geometry.
    x0 = max(0, int(np.floor(vertices[:,0].min())))
    y0 = max(0, int(np.floor(vertices[:,1].min())))
    x1 = min(component.shape[1], int(np.ceil(vertices[:,0].max()))+1)
    y1 = min(component.shape[0], int(np.ceil(vertices[:,1].max()))+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    crosses=[]
    for i in range(len(vertices)):
        a,b=vertices[i],vertices[(i+1)%len(vertices)]
        crosses.append((b[0]-a[0])*(yy-a[1])-(b[1]-a[1])*(xx-a[0]))
    region = np.all(np.array(crosses)>=-1e-8,axis=0) | np.all(np.array(crosses)<=1e-8,axis=0)
    if shape == 'circle':
        region = (xx-apex[0])**2+(yy-apex[1])**2 <= (altitude/2)**2
    plausible=(hsv[:,:,0]>=85)&(hsv[:,:,0]<=140)&(hsv[:,:,2]>=20)
    sats=hsv[:,:,1][plausible]
    floor=max(65,float(np.percentile(sats,90))*.65) if sats.size else 65
    color=hsv[y0:y1,x0:x1]
    blue=(color[:,:,0]>=85)&(color[:,:,0]<=140)&(color[:,:,1]>=floor)&(color[:,:,2]>=20)
    return dict(geometry=shape, theta_degrees=theta,
                altitude_pixels=altitude, vertices=vertices.tolist(), base_center=base.tolist(),
                complete_triangle=complete, sampled_pixels=int(region.sum()),
                blue_fraction=float(blue[region].mean()) if region.any() else 0.0)


def inspect(sheet, surround_threshold=None, radius_fraction=.02, geometry='ring', large_exception=True):
    """Mark light spots using each crop's measured blue silhouette; require review."""
    tiles = corner_tiles(sheet)
    overlay = sheet.copy()
    candidates, corners = [], []
    excluded_candidates = []
    source_candidate_id = 0
    for name, offset_x, offset_y, crop in tiles:
        band, interior = inspection_band(name, crop)
        view = overlay[offset_y:offset_y+crop.shape[0], offset_x:offset_x+crop.shape[1]]
        outline = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_GRADIENT,
                                   np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
        if surround_threshold is None:
            view[outline.astype(bool)] = (255, 255, 0)
        measured = cv2.GaussianBlur(crop, (3, 3), .6)
        lab = cv2.cvtColor(measured, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(measured, cv2.COLOR_BGR2HSV)
        light = lab[:, :, 0].astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32)
        baseline = cv2.medianBlur(lab[:, :, 0], 31).astype(np.float32)
        sat_baseline = cv2.medianBlur(hsv[:, :, 1], 31).astype(np.float32)
        pale = (light-baseline > 13) & ((sat_baseline-saturation > 12) | (light-baseline > 30))
        blue = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 140) & (saturation > 45))
        # Select neighborhoods containing blue ink, without constructing an edge.
        near_blue = cv2.dilate(blue.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        mask = (pale & near_blue).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        ids = []
        for index in range(1, count):
            x, y, width, height, area = stats[index].tolist()
            if area < 3:
                continue
            source_candidate_id += 1
            component = labels == index
            overlap = int(np.count_nonzero(component & band))
            interior_overlap = int(np.count_nonzero(component & interior))
            scaled_area = area * (500/crop.shape[0])**2
            ring_metrics = {}
            if surround_threshold is not None:
                if geometry in ('triangle','square','circle'):
                    ring_metrics = triangle_blue(component,hsv,name,radius_fraction,geometry)
                    enclosed = (ring_metrics['complete_triangle'] and ring_metrics['sampled_pixels']>0
                                and ring_metrics['blue_fraction']>=surround_threshold)
                    if geometry == 'circle':
                        yy,xx=np.where(component)
                        cv2.circle(view,(round(float(xx.mean())),round(float(yy.mean()))),
                                   round(crop.shape[0]*radius_fraction/2),(255,255,0),1)
                    else:
                        cv2.polylines(view, [np.rint(ring_metrics['vertices']).astype(np.int32)],
                                      True, (255,255,0), 1)
                else:
                    ring_metrics = blue_surround(component, hsv, radius_fraction)
                    enclosed = (ring_metrics['complete_ring'] and
                                ring_metrics['blue_fraction'] >= surround_threshold and
                                ring_metrics['minimum_sector_blue_fraction'] >= .35)
                if enclosed and (not large_exception or scaled_area < 100):
                    code = {'triangle':'T', 'square':'S', 'circle':'C', 'ring':'R'}[geometry]
                    excluded_candidates.append(dict(id=source_candidate_id, corner=name,
                        reason_code=code, display_label=f'{source_candidate_id}{code}',
                        area_at_500px=scaled_area, surrounding_blue=ring_metrics,
                        box_in_corner=[x,y,width,height], reason='surrounded_by_blue'))
                    cv2.rectangle(view, (x,y), (x+width,y+height), (0,180,0), 1)
                    cv2.putText(view, f'{source_candidate_id}{code}', (x,max(12,y-3)),
                                cv2.FONT_HERSHEY_SIMPLEX,.4,(0,180,0),1)
                    continue
                retained_by = 'large_mark_exception' if enclosed else 'not_enclosed_by_blue'
            elif overlap >= area*.5:
                retained_by = 'outside_inner_boundary'
            elif scaled_area >= 100 and (interior_overlap >= area*.5 or overlap > 0):
                retained_by = 'large_interior_mark_exception'
            else:
                continue
            candidate_id = source_candidate_id
            code = 'L' if retained_by in ('large_mark_exception','large_interior_mark_exception') else ''
            ids.append(candidate_id)
            candidates.append(dict(id=candidate_id, corner=name, category='light_spot_candidate',
                                   reason_code=code, display_label=f'{candidate_id}{code}',
                                   surrounding_blue=ring_metrics,
                                   retained_by=retained_by, area_at_500px=round(scaled_area, 2),
                                   fraction_in_band=round(overlap/area, 3),
                                   box_in_corner=[x, y, width, height],
                                   box_in_sheet=[x+offset_x, y+offset_y, width, height], pixels=area,
                                   local_brightness_increase=round(float(np.mean((light-baseline)[component])), 2)))
            view = overlay[offset_y:offset_y+crop.shape[0], offset_x:offset_x+crop.shape[1]]
            cv2.rectangle(view, (max(0, x-2), max(0, y-2)),
                          (min(crop.shape[1]-1, x+width+2), min(crop.shape[0]-1, y+height+2)), (0, 0, 255), 1)
            cv2.putText(view, f'{candidate_id}{code}', (max(0, x-2), max(12, y-3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
        corners.append(dict(corner=name, source_box=[offset_x, offset_y, crop.shape[1], crop.shape[0]],
                            candidate_ids=ids, boundary_status='measured' if band.any() else 'unconfirmed'))
    return dict(status='needs_review', candidate_count=len(candidates), candidates=candidates, corners=corners,
                excluded_candidates=excluded_candidates,
                input_mode='existing_corner_tip_sheet', measurement_image_size=[sheet.shape[1], sheet.shape[0]],
                band_calibration='Each crop measured independently; inner cutoff 2.4% of tile width; no outer cutoff',
                large_interior_min_area_at_500px=100,
                limitations=['Light spots may be glare, dust, background or damage',
                             'Enlarged sheet pixels are not independent original-photo pixels',
                             'No automatic clean or damaged verdict', 'Folds are not measured']), overlay
