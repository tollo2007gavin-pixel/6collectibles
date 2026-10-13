# eBay combined yellow-border runner

Open `ebay_line_fit_test.py` in VS Code, select the Proj001 Python environment, and choose **Run Python File**. Paths resolve relative to the script. If needed, run `python -m pip install -r requirements_line_fit.txt`.

The runner uses existing `EBAY_CLIENT_ID` and `EBAY_CLIENT_SECRET` from Proj001's `.env`. No OpenAI key or vision API is used. Never put secrets in the script.

## Batch controls

Edit the settings at the top. Your current settings are preserved: 45 total fronts, 3 listings per card, 60 inspected candidates per card, and no listing-age restriction.

- `TOTAL_CARDS_ANALYZED`: maximum downloaded fronts processed, including unresolved fits.
- `MAX_LISTINGS_PER_CARD`: maximum processed listings for each active CSV row.
- `MAX_CANDIDATES_PER_CARD`: search-result inspection budget; nonqualifying listings count here.
- `LISTING_AGE_HOURS`: `None` for any age, or e.g. `24` for the past day.
- `SHUFFLE_CARDS` / `RANDOM_SEED`: query order and optional repeatability.
- `EXCLUDE_PREVIOUS_BATCH`: skip listing IDs from the immediately preceding run.
- `CARD_RARITY_FILTERS`: CSV ID 39 (Ditto GG22/GG70) and ID 28 (Turtwig GG31/GG70) require Rarity = Ultra Rare. Other card searches have no rarity restriction.
- `MEASUREMENT_FRAME`: defaults to `refit_when_supported`, using the validated rebuilt frame when available.
- `PERSPECTIVE_LIMIT`: defaults to `0.20`. Other physical and boundary checks remain active.
- `ANGULAR_REVIEW_THRESHOLD`: defaults to `0.15`; measured axes above this receive the proposed +/-1-point angular allowance.

Optional terminal overrides: `python ebay_line_fit_test.py --total-cards 10 --max-listings-per-card 2`.

Only `active=1` rows from `data/cards.csv` are searched. Only ungraded listings with structured Card Condition **Near mint or better** qualify. Rarity uses the seller's structured item aspect, not a title keyword; listings missing that value may be excluded. The runner downloads only the main listing image, assumed to be one card's front. It never downloads additional images. Seller metadata and keyword matches do not independently verify identity or grade.

## One output image per listing

Both existing fitters still run: adaptive (straight or curved) and forced curve. The runner creates only one JPG in `centering_output/ebay_line_fit/Combined/`:

1. For each of eight borders, display forced curve if available; otherwise display the adaptive path. If neither is supported, leave that border blank.
2. For each measurement axis independently, average the adaptive and forced-curve percentages when both pass their existing checks.
3. If only one method has a valid measurement for that axis, use it. If neither qualifies, leave the value unresolved.

This averages reported ratios, not point coordinates or raw border widths. The complementary side is 100 minus the averaged percentage. The header identifies whether each axis uses a mean or a single method; its number need not equal the measurement of the displayed forced-curve path alone. Averaging is not proof of improved accuracy: the methods share their frame and measurement calculation.

The pipeline retains support, resizing, locality, consensus and physical checks. Paths may be shown diagnostically when a physical check withholds margins. Adaptive is not a strictly straight-only reference. Curve-fitting thresholds stay unchanged; the separate image-quality gate below adds a hard rejection for missing exterior background.

### Missing interior-edge recovery

If the existing interior fitter cannot retain an edge, the pipeline now tries brightness alongside the existing colour contrasts. A second sampling span excludes the outermost 9% at each end to avoid printed labels that overlap the border. For glare/foil texture it can also smooth along the side before detecting transitions. These fallbacks are shared across cards; there are no target-centering values or card-specific coordinate corrections.

Recovery runs only on frames that pass the existing physical checks. Each proposal must still pass the existing support, endpoint distribution, local search and half-resolution stability checks. At least two profile variants must agree within one source pixel over their common span. The selection favours a longer span and less blur. Measurements use only the common observed span of all four relevant edges, without extending shortened paths to the corners. Already accepted interior fits and exterior fits are preserved; failed geometry and missing-background gates still withhold measurements.

### Pale yellow border validation

A frame rejected only by the yellow-colour checks can now be measured when one side is desaturated but still has consistent yellow hue. This recovery requires all eight existing tracks to have at least 90% support and pass resizing stability; three sides must independently classify as yellow over at least 80% of their sampled bands. The fourth must retain yellow hue over at least 90% of its band, agree with the other sides' hue, and have substantially lower saturation. Both its artwork and background transitions must show contrast across at least 85% of the sampled span. Samples come from the retained paths on the original image; no image recolouring, boundary relocation, card-name exception or centering target is used.

This overrides only colour rejection reasons. Perspective, physical geometry, missing-edge and exterior-background gates still apply. Per-trial `colour_recovery` diagnostics preserve the original reasons and sampled evidence. Neutral grey holder strips, multiple pale sides and weak/missing paths do not qualify through this rule.

### Hard fail: missing exterior background

Images now receive **Fail** when the source-image audit finds at least three of four exterior sides without sufficient surrounding background. Fail suppresses usable L/R and T/B values in both methods and the combined result, even if some diagnostic lines can be drawn. JSON records `image_gate.fail`, the code `insufficient_exterior_background`, side names and evidence; the image header and CSV show Fail. This is a capture-quality rejection, not a grading or centering verdict.

The audit inspects the central 70% of each image side. It looks for a narrow yellow strip continuing to the image boundary (supported on at least 80% of sampled positions, transitioning into artwork within 10% of image depth), or an exterior frame lacking a 3-source-pixel outward strip over at least 80% of the side. Retained exterior tracks can establish visibility only when the image also supports a yellow-inside/non-yellow-outside transition. A large yellow background alone does not prove clipping. One or two unavailable sides, ordinary missing fits and uncertain evidence remain Review/unresolved under the existing rules.

This catches the reviewed edge-to-edge stock artwork and tightly cropped photos, but is not a universal stock-image classifier. A stock image with a convincing surrounding background may need a separate detection mechanism. It does not treat missing fitter output as proof of a missing physical edge. Side names here are image-facing directions, not text orientation. Three-pixel clearance is a practical sampling requirement, not a claim that every such side was cropped by a specific number of pixels.

`review.html` shows the combined images. `measurements.csv` contains one combined row per listing; `batch.json` and per-image JSON in `details/` retain both original methods, combined measurements and path provenance. `downloads/` contains full-resolution, EXIF-oriented working images.

## Batch replacement

Each run clears the runner's `Combined`, `downloads` and `details` folders and replaces its reports. Legacy `Forced_curve` and `adaptive` folders within this same batch directory are also cleared during migration and are no longer generated. Other reviews and regression archives are untouched.

CSV validation and successful authentication happen before cleanup. Zero-result runs produce an empty batch. A `.owner` marker and exclusive `.running` lock protect cleanup; links/junctions are refused. If a killed run leaves `.running`, verify the runner is stopped before removing that single lock file. Do not store personal files in the automatically cleared folders.

## Perspective-frame candidates and measurement stability

Each fitted trial now records an advisory `frame_diagnostics` block. It rebuilds
an outer quadrilateral from straight supporting lines fitted to the final outer
paths, then samples the same curved borders through that candidate frame.
It requires all four retained outer paths and checks convexity, orientation and
a 2 mm corner-locality bound. These construction checks do not reject the image.

`MEASUREMENT_FRAME = 'refit_when_supported'` is now the eBay runner default.
It selects rebuilt frames for already-qualified axes;
an unavailable candidate falls back to the initial frame. Curved border paths
are preserved. Existing physical, coverage, stock/crop and perspective gates
still apply, with the runner's configured 20% perspective limit. Candidate margins for rejected
photos are diagnostics, not accepted measurements.

The CSV adds candidate-frame percentages, signed changes from the initial frame,
and worst absolute half-resolution shifts across the contributing fit methods.
Candidate numeric CSV fields stay blank for rejected axes. The detailed JSON
separates path-only, frame-only and combined resizing changes, records source
border-width pixel evidence, and keeps missing checks unavailable rather than
calling them stable. The checks are per axis; no missing side is silently used
as measured evidence. The CSV's frame-policy column identifies active selection.

The runner labels complete measurements as `pass` through perspective 0.15,
or `pass/review` above 0.15 through 0.20 with a proposed +/-1 percentage-point
allowance on affected axes. JSON retains ranges and per-axis perspective;
CSV and overlays show the category and allowance. `fail` remains the hard
exterior-background rejection, and incomplete measurements are `undetermined`.
These labels describe measurement eligibility, not acceptable card centering.
The serial runner scopes the perspective override to each analysis call;
direct engine callers retain their existing defaults.

No resizing-based uncertainty cutoff is enabled. Resampling differences are stress tests,
not calibrated error bounds; agreement of the two related fits cannot rule out
shared bias. A straight supporting frame is distinct from the curved paths used
to measure border widths. Scans and holder repeatability still need separate
consideration when interpreting the proposed angular allowance.

### Future centering gate using estimated ranges

`yellow_batch_engine.measurement_screening.screen_axis` implements the requested
overlap policy, but is not enabled in the runner. It takes an externally supplied
uncertainty width; it does not turn resampling shifts into confidence intervals.
At a maximum-side gate of 52.5%, both sides must be between 47.5% and 52.5%.
L=53% with an estimated +/-0.5 pp range touches that allowed interval and is
accepted with `accepted_due_to_uncertainty`. R is the exact complement, not an
independent interval. A point estimate inside the gate and an entire interval
inside the gate receive distinct provenance. Existing measurement-quality
failures still withhold the card regardless of the proposed interval. Missing
uncertainty is unavailable, not an assumed zero. No confidence level is claimed.

This policy intentionally keeps borderline candidates for later assessment.
Range widths and the actual centering cutoff need calibration/configuration
before activating the screen. The runner now records the proposed angular
allowance described above but does not apply any centering selection gate.

## Offline checks

### Background-contamination fallback

When every margin is withheld because the starting physical frame failed,
the yellow engine can try alternative localization masks that remove thin
connections to similarly coloured background patterns before filling holes.
The replacement must be supported by at least two agreeing mask variants,
pass the existing physical checks, and supply valid source-pixel fits for
both measurement axes. It never retries an existing usable axis or hard
exterior-background failure. Cleaning affects localization masks only;
measurement profiles always use the original image. There are no card-name,
listing-ID, coordinate, or target-centering exceptions.

The configured perspective limit and measurement-frame policy are unchanged.
Geometry can be repaired while measurements remain withheld by that limit.
Use `compare(..., recover_background=False)` for a diagnostic baseline.

The September 22 review covers 144 cached images (118 eBay listings and 26
Mareep images). The two recovered listings exceed the former runner
perspective limit of 0.10; the validated 0.20 policy is now enabled in the runner.
See `centering_output/background_recovery_review`
for before/after overlays and the audit. Other staged recovery categories are
not included in this change.

```powershell
python -m unittest test_ebay_line_fit -v
python -m unittest test_interior_recovery -v
python -m unittest test_pale_border -v
python -m unittest test_frame_diagnostics -v
python -m unittest test_measurement_screening -v
python -m unittest test_background_recovery -v
python -m unittest test_edge_recovery -v
python -m unittest test_lighting_recovery -v
python -m unittest test_boundary_selection -v
python ebay_line_fit_test.py --local-images test_images/IMG_7955.png
```

Local mode makes no network requests and uses a separate `centering_output/ebay_line_fit_local/` folder. The isolated `yellow_batch_engine` remains separate from production `cardcv` and the AI pregrade script.

eBay references: [condition IDs](https://developer.ebay.com/api-docs/sell/static/metadata/condition-id-values.html), [trading-card condition descriptors](https://developer.ebay.com/api-docs/user-guides/static/mip-user-guide/mip-enum-condition-descriptor-ids-for-trading-cards.html), [aspect filtering](https://developer.ebay.com/api-docs/buy/api-browse.html#filtering-by-item-aspect).

## Interrupted or unstable edge recovery

Missing exterior tracks can now be recovered from agreeing BGR, chroma, or luminance profiles. Recovery varies along-edge smoothing and trims up to 12% at each end, while retaining the existing 75% support, distributed-coverage, locality, and half-resolution checks. At least two variants must agree within one source pixel over their common observed span. Interior recovery first preserves its existing searches, then tries the additional central span. No card identifiers, target margins, or widened normal search are used.

Accepted paths are preserved. A newly recovered exterior that conflicts with an accepted interior is discarded. Measurement still uses the shared observed span, never inferred endpoints of these shorter tracks. This fallback adds processing time only to eligible missing edges; it does not override geometry, perspective, colour, or exterior-background rejections. It respects the calling runner's perspective and frame policy.

See centering_output/edge_recovery_review for before/after overlays, close-ups of newly retained edges, and validation results.

## Lighting variation and border colour

A chroma-only rejection can now be rechecked against the already retained boundary tracks. The original chroma statistic mixes hue and saturation, so a washed-out but still yellow strip can disagree with the other sides. Recovery requires all eight boundaries to have strong distributed, resize-stable support; at least 85% yellow samples on each side; matching side hues; and distinct transitions to both the artwork and background. It uses original source pixels and never moves lines or changes a measurement frame.

The fallback cannot override missing-yellow, perspective, geometry, or crop rejections. The older one-pale-side recovery is preserved. No card identities or desired margins are used. See centering_output/lighting_recovery_review for the recovered Snorlax, colour evidence, production and experimental margins, and the 144-image audit. The six saved Snorlax listings were rerun; only 13_445868020f5ea7b0 changes acceptance. Severe glare that removes colour or boundary evidence remains unresolved.

## Wider borders competing with printed transitions

This is a recovery step after the existing pipeline rejects a physical frame. Any already usable axis, hard exterior-background failure, or otherwise valid frame rejected only by perspective is preserved. The recovered frame must use an added wider pair, have agreement across at least two localization variants within 1.5 percentage points, and retain qualified boundary measurements on both axes. The configured perspective policy still applies. `compare(..., recover_boundaries=False)` disables this recovery for diagnostic comparisons.

The boundary-pair selector now reconsiders already detected edge pairs up to its existing relaxed 4.2 mm width limit when the nominal-width candidates lack convincing physical-border evidence. Previously, any nonempty pair list suppressed a relaxed retry, even if all remaining pairs were printed lines inside the artwork. An actual wide border just above the nominal 3.4 mm cutoff could therefore be excluded while the printed alternatives survived. These mm values use a local image scale and are not direct physical measurements.

Only added pairs with at least 85% yellow inside, at most 25% yellow immediately outside, and at least 50% mean row support enter the candidate ranking. The detected peaks, normal search extent, width penalty, frame consensus and downstream fitting/measurement gates are unchanged. A supported nominal pair suppresses this retry. No opposite-border equalization, card-specific parameters or target ratios are introduced. Wider borders beyond the existing relaxed limit still need other evidence and are not accepted automatically.

See centering_output/boundary_selection_review for the recovered Snorlax, traced pair-selection decision, and the full saved-image regression results.
