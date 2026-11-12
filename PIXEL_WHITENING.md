# Experimental four-corner inspection

## Optional edge uncertainty strategy

`--edge-filter` adds a conservative background gate before grain and geometry. `edge_uncertainty.py` creates three blue silhouettes using saturation factors .55/.65/.75, rejects implausible or strongly disagreeing estimates, and protects the union with a 4%-of-tile margin. A connected candidate is excluded only if all its pixels are outside the protected region. Bright neutral components touching the edge are treated as uncertain. This deliberately retains more candidates against white/glared backgrounds. It does not claim exact outlines, calibrated uncertainty, or reliable dust/damage separation. The grain/triangle stages remain unchanged and may still reject a genuine chip.

`B` means background rejection; `G` and geometry letters retain their prior meanings. An unconfirmed edge skips the B filter. Original pixels remain unchanged; amber shading exists only in the audit overlay. Saved candidate IDs and source hashes make baseline comparisons possible. No large-spot exemption has been reintroduced.

Offline evaluation tools: `evaluate_edge_filter.py` measures margin sensitivity and known-example perturbations; `verify_edge_replay.py` checks exact legacy decisions and repeated enhanced decisions against saved JSON. These are regression checks, not independent accuracy evaluations. The saved fallback tag is pixel-fallback-before-edge-uncertainty.

Optional grain filtering (`--grain-filter`) runs before geometry, preserving original candidate IDs and recording G exclusions. It estimates local brightness-residual noise using median absolute deviation in nearby blue pixels within the estimated card interior, excluding each candidate and a small margin. This is not a full validated segmentation of the area between the outer card edge and the interior artwork rectangle. Border/artwork contamination remains possible.

The conservative initial rule requires at least 100 reference pixels, noise sigma between 4 and 18, candidate area below 40 normalized pixels, 75th-percentile brightness residual below a noise-dependent threshold (13–30), and 75th-percentile saturation loss below 30. Large or strongly contrasting marks survive. Insufficient or unstable reference skips grain rejection; overall output remains needs_review. Glare classification and damage accuracy are not established by this rule. Compare G labels against human-reviewed chips before increasing its strength.

Run `python pixel_review.py "reviewed.html" --out "new-output-folder"` to inspect the saved Corner close-ups images in a portable reviewed HTML. The output folder must be new. Outputs include an embedded comparison gallery, annotated sheets, and JSON measurements. It uses no model calls.

The input is the exact existing 2 by 2 corner-tip sheet made by `card_utils.corner_closeup_sheet`. Its four tiles are read without resizing, warping, or running the card detector. Captions and separators are excluded. Source hashes and per-corner coordinates are recorded. A card with no saved corner sheet is explicitly skipped. Unknown layouts raise an error.

Red boxes mark local clusters brighter than their neighborhood near blue pixels. These are light-spot candidates, not confirmed chips. All decisions remain `needs_review`. Glare, dust, and textured backgrounds can still trigger candidates; refining those distinctions is deferred. There is no whole-edge inspection or fold classifier. Counts refer to enlarged sheet pixels rather than independent source-photo pixels. Back colors other than blue require separate calibration.

Each corner independently supplies a blue-ink interior mask, using relative saturation to reduce sleeve contamination, morphological opening and a convex hull. The inward crop corner is included to prevent printed artwork from making an internal boundary. Cyan shows only the inner cutoff, 2.4% of tile width inward (12 pixels at a 500-pixel tile). There is no outer cutoff within the saved crop. This is a provisional width, not a learned damage standard. Crop-frame edges are ignored using replicated padding. This segmentation remains approximate and can fail on glare, shadows or blue backgrounds.

The outline mode requires at least half the candidate pixels outward of the inner cutoff. Geometry modes exclude candidates that meet their blue threshold regardless of area. There is no large-spot exception and no L label. Grain filtering remains limited to small weak spots; larger spots still undergo the full geometry test.

Human corrections are shown alongside measurements, never used as detector inputs. No detection accuracy should be inferred from candidate counts. Tests exercise the real sheet layout, caption exclusion, corner attribution, source preservation, repeatability, and rejection of incompatible layouts. Run `python -m unittest test_pixel_whitening`.
