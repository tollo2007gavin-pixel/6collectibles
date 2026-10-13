"""Future centering screen using explicitly supplied measurement ranges.

No uncertainty is inferred from resizing diagnostics here. This helper is not
enabled in the eBay runner and must not bypass existing measurement-quality gates.
The range has no statistical confidence level unless independently calibrated.
"""
import math


def screen_axis(percentage,uncertainty_pp,*,max_side_pct,measurement_qualified=True):
    """Keep a card if its possible centering intersects the allowed range.

    `percentage` is L% (or T%); the opposite margin is exactly 100-percentage.
    `max_side_pct=52.5` allows L% in [47.5,52.5], rather than admitting all
    left-light cards. Equality at the cutoff counts as a possible pass.
    """
    if not math.isfinite(max_side_pct) or not 50<=max_side_pct<=100:
        raise ValueError('max_side_pct must be a finite percentage from 50 to 100')
    result=dict(accepted=False,range_kind='provided_estimate',statistical_confidence_level=None,
                allowed_percentage_range=[100-max_side_pct,max_side_pct])
    if not measurement_qualified:
        return dict(result,status='withheld_unqualified_measurement')
    if uncertainty_pp is None:
        return dict(result,status='withheld_unavailable_uncertainty')
    if percentage is None or not math.isfinite(percentage) or not 0<=percentage<=100:
        raise ValueError('percentage must be finite and from 0 to 100')
    if not math.isfinite(uncertainty_pp) or not 0<=uncertainty_pp<=100:
        raise ValueError('uncertainty_pp must be finite and from 0 to 100')
    low=max(0.,percentage-uncertainty_pp);high=min(100.,percentage+uncertainty_pp)
    allowed_low,allowed_high=result['allowed_percentage_range']
    overlap_low=max(low,allowed_low);overlap_high=min(high,allowed_high)
    point_passes=allowed_low<=percentage<=allowed_high
    accepted=overlap_low<=overlap_high
    fully_inside=allowed_low<=low and high<=allowed_high
    if not accepted:status='rejected_outside_estimated_range'
    elif fully_inside:status='accepted_within_estimated_range'
    elif point_passes:status='accepted_point_estimate'
    else:status='accepted_due_to_uncertainty'
    return dict(result,accepted=accepted,status=status,percentage=percentage,uncertainty_pp=uncertainty_pp,
                percentage_range=[low,high],opposite_percentage_range=[100-high,100-low],
                point_estimate_passes=point_passes,range_fully_within_gate=fully_inside,
                possible_passing_range=[overlap_low,overlap_high] if accepted else None)
