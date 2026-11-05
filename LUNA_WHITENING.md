# Reviewed Luna whitening pipeline

This is the optimized whitening-only configuration used in the latest human-reviewed 20-card batch. It does not run the separate yellow-border/centering pipeline.

## Results and limits
Your review matched 17 of 19 definite predictions (89.5%): all 11 whitening cards were flagged, and 6 of 8 clean cards were accepted. One uncertain card was excluded. You marked 19 of 20 crops aligned. This is one small, selected batch with predictions visible during review, not proof of consistent 90% accuracy. The original model run made 43 calls, estimated at $0.0499 using its configured rates. Future pricing and token usage may differ.

## Setup
Install requirements.txt and configure OPENAI_API_KEY and the eBay credentials described in SETUP.md in a local .env. Never commit that file. The optional requirements-audit.lock records the Windows environment used during testing.

The 16 user-confirmed calibration photos and labels are included in references/luna/ and luna_references.json with the user's explicit permission. The bank contains eight clean and eight not-clean examples; labels come from the user's explicit review, not original filenames. Each file is protected by its recorded SHA-256 hash. The example manifest is only a template for replacing the bank; the included real manifest is ready to use.

Each target receives eight balanced other-card examples. SHA-256 checks prevent unnoticed changes to reference images. Relative paths resolve from the module directory; absolute local paths are also supported. Without a valid local calibration bank, Luna refuses to run.

## Prepare automatic crops
Organize downloaded eBay results into a NEW folder:

```sh
python organize_listings.py --from-results out/results.jsonl --out cards_luna --whitening-boundaries
```

Use the statuses option if the fetcher's output uses other statuses. The boundary option is deliberately opt-in. It refines back photos only, saves original bytes and green boundary overlays, and sets ambiguous photos to card_found=false. That means the crop is unvalidated, not that the listing lacks a back. Do not use that crop as condition evidence; the runner retains the source for review and reports a sheet-preparation failure. Existing output folders are refused to prevent mixing old inspection sheets with new crops.

For a photo whose manifest boundary_review.status is ready, produce the ordinary sheet and corner tips from its normalized image using card_utils.edge_corner_sheet(image, 'back') and card_utils.corner_closeup_sheet(image, 'back'). Save the tips as PNG and pass the matching original_file as context. Do not substitute sheets from a different photo. The source-image preparation option in WHITENING_INPUTS.md is an alternative experimental input format, not the layout validated for this Luna result.

## Inspect one card

```sh
python luna_whitening.py --context cards_luna/CARD_ID/original_back_1.jpg --sheet sheet_1.jpg --tips tips_1.png --card-id CARD_ID --out outputs/result.json
```

Without --run this checks inputs and references without making API calls. Add --run to perform the paid assessment. The original may instead be PNG or WebP; use the manifest's actual filename. This CLI expects already validated matching crops; it does not independently run the boundary detector.

The configured model is gpt-6-luna with xhigh reasoning. It makes two inspections, then a third if their decisions differ. Allowed whitening slips are zero. Invalid responses or API failures remain needs_review. Results include each response and token usage. Model availability and access depend on the API account.

## Offline tests

```sh
python -m unittest discover -p "test_*.py"
```

75 tests passed in the upload checkout, with no paid API calls. The yellow_border directory and card_utils (1).py were preserved from the latest remote branch. card_utils.py supplies the imports required by the root whitening pipeline. Generated reviews, listing batches, local notes, and credentials are not part of this upload.


## Parallel inspections and timing
The first two independent inspections run concurrently by default. A third runs only after both valid results arrive and disagree. Stage order stays standard view, tip view, tie-breaker regardless of completion order. Use --sequential for a comparison with the original ordering.

Result timing includes file-hash validation, request preparation, concurrency-slot wait, each API call, and total assessment time. API duration includes transport and provider processing; it does not isolate server reasoning. These timers cover classification, not upstream eBay downloading or crop detection. Overlapping durations must not be added to estimate wall-clock time.

A shared three-request cap per Python process prevents nested batch concurrency from doubling simultaneous API traffic. This can shorten individual-card latency, but a batch already keeping three requests busy may see little throughput gain. Multiple processes have separate caps.

If one initial call fails or returns an invalid result, the other may already be running. Both outcomes and available usage are retained; no tie-breaker is sent, and the card remains needs_review. Parallel operation cannot promise the sequential path's one-call cost when the first response fails. No automatic retries are added.

## Current GitHub runner: mandatory whitening sheets

Every paid whitening assessment requires one original back, the four-corner and four-edge sheet, and a corner-tip close-up sheet. The classifier refuses original-only input before issuing an API call. The standard and tip inspections retain the original reviewed prompt and reference bank; timing and parallel execution remain enabled.

The original-photo fallback introduced on November 1 has been removed after human review exposed missed chips. When a sheet cannot be prepared, the runner returns `back_corner_sheet_unavailable` / Needs review and preserves the original photos. It does not claim that the listing lacks a back or force a Clean/Not clean decision. A missing or unreadable image remains a separate preparation failure.

Boundary detection retains successful legacy geometry. Failed detections retry line searches close to the seeded perimeter (reducing background/artwork clutter), then a more tolerant pale-blue color profile. A wider search also handles seeds on the inner artwork panel. Every accepted outline still needs supported outer lines on all four sides and the existing shape/cut-off checks. Previously unknown-side photos are reconsidered only when the recovered outline has blue support on all four sides and the normalized image classifies as a back. These changes are whitening-only; centering modules are unchanged.

Offline replay of the November 5 reviewed batch prepares valid sheet packets for nine of ten listings, versus one originally. The tenth photo is cut off at the top and is withheld. This is image-preparation coverage, not a new accuracy measurement. No API judgments were rerun or attached to the new sheets. The old 91.9% result does not establish accuracy for newly recovered crops; their alignment and model results require fresh human review.
