# Run the card pipelines yourself

**One entry file: `run_cards.py`.** Edit its `SETTINGS` block and press Run in VS Code,
or use the GitHub Actions form. The same production code runs in either place.
The runner uses the reviewed Luna pipeline and current yellow-border pipeline;
the older OpenCV whitening scores and alternative provider experiments are not used.
It also includes the [guarded GX pipeline](gx_border/README.md). Choose
`border_type="yellow"`, `"gx"`, or `"all"`. `all` selects both approved catalogs;
it never applies both fitters to the same card.

## Run directly on GitHub

1. In this repository, open **Settings → Secrets and variables → Actions**.
   Create repository secrets `EBAY_CLIENT_ID`, `EBAY_CLIENT_SECRET`, and `OPENAI_API_KEY`.
   Existing secrets with those exact names can be reused. Do not paste keys into code.
2. Open **Actions → Run card pipeline → Run workflow**.
3. Choose modes, listing limits, time window, card IDs, centering gate and API cap.
4. When finished, open the workflow run and download **card-results-N** under Artifacts.
   Extract the ZIP, then open the run's `review.html`. Keep its `_audit` folder beside it
   so the testing images load. Results are retained on GitHub for 14 days.

This workflow is **manual only**. Pushing changes does not search eBay or spend API money.
GitHub must have Actions enabled; the workflow must be on the default branch to show
the Run workflow button. Repository collaborators need permission to run workflows.
GitHub runner minutes are separate from whitening API charges and depend on your plan.
Runs are serialized; they do not cancel an earlier run. The workflow has a two-hour timeout.
Select `check_only` to verify installation, settings and reference hashes without
fetching listings or calling the API. A check-only run produces logs, not a results artifact.

The Actions form overrides matching values from `run_cards.py`; other defaults come
from that file. Workflow defaults are explicit, so update the form too if you want to
change its default choices. GitHub runs start on fresh machines: cross-run skipping
and local resume are intended for your computer, not automatic artifact downloading.

## Run on your computer / VS Code

Use Python 3.12+ and a dedicated virtual environment from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements_runner.txt
Copy-Item .env.example .env
# Edit .env and enter your keys. Select .venv as VS Code's Python interpreter.
.venv\Scripts\python run_cards.py --check
.venv\Scripts\python run_cards.py
```

Copy `.env.example` only for initial setup; keep your existing `.env` if configured.
`--check` validates files/settings/reference hashes without fetching or paid calls,
and lists missing credential names. Centering-only does not need an OpenAI key.
Local-photo runs do not need eBay keys. `prepare_api_only=True` does not need an OpenAI key.
All relative file paths resolve from the repository, so Run works from other folders.

## Modes and gates

| Setting | Meaning |
|---|---|
| `border_type="yellow"` | Only the 15 approved active yellow designs. |
| `border_type="gx"` | Only the nine approved active GX designs in gx_border/data/cards.csv. |
| `border_type="all"` | Both lists, each card dispatched to its assigned engine. Limits count the combined batch. |
| `pipeline="both"` | Centering first. Only complete centering passes proceed to validated back preparation and Luna. |
| `pipeline="centering"` | Front measurements and centering gate only. No whitening API calls. |
| `pipeline="api"` | Back whitening only. No centering analysis or centering gate; an accepted card is **not** certified centered. |
| `output_mode="production"` | Main report/CSV/JSON list only accepted cards, with title, listing link, margins when measured, and brief whitening evidence. |
| `output_mode="testing"` | Main report includes every analyzed listing, front overlay when available, usable original back, reasons, and whitening evidence. |
| `centering_max_side_pct=53` | Both axes must be between 47 and 53. Setting 52.5 means 47.5–52.5. |
| `allow_angular_variance=True` | Yellow only: eligible perspective scores above 0.15 up to 0.20 have a proposed ±1-point allowance. GX keeps its own quality checks and no added allowance. Report marks passes that depend on the allowance. Set False for a strict point-estimate gate. |
| `prepare_api_only=True` | Prepare/inspect back crops without paid API calls. In `both`/`api` these results remain undetermined, not accepted. |

The quality gates are retained: hard exterior-background failures, unresolved axes,
and excessive perspective cannot pass centering. Luna's ambiguous back crops,
`needs_review`, errors and invalid responses never pass whitening. The allowance
is an operating policy, not a statistically calibrated confidence interval.

Whitening retains the reviewed model, prompt, references, and two inspections plus
an adjudication call on disagreement. A rejected listing is never repeatedly assessed
to get a favorable answer. This checks visible back whitening, not authenticity or a grade.

## Fetching, limits and useful controls

| Setting | Default | Meaning |
|---|---:|---|
| `max_listings` | 45 | Maximum eligible listing records analyzed, including processing errors. It is not the pass target. |
| `max_listings_per_card` | 3 | Per active CSV query. A listing has one presumed front: its first image. |
| `max_listing_age_hours` | None | Maximum time since eBay listing start; None = any age. This is not the time the physical card has ever been offered across relistings. |
| `max_candidates_per_card` | 100 | Search/detail budget per query. Non-NM listings can consume candidates without consuming analysis slots. |
| `card_ids` | [] | All active approved rows in the chosen border family, or selected IDs. Yellow examples: `["22", "23"]`; GX: `["eevee_sm233", "pikachu_sm232"]`. |
| `cards_csv` | yellow_border/data/cards.csv | Approved yellow queries; active rows only. Existing Ditto/Turtwig Ultra Rare filters are preserved. |
| `gx_cards_csv` | gx_border/data/cards.csv | Separate approved GX queries. Yellow's per-card rarity filters never apply to GX. Both families retain structured Near Mint-or-better/ungraded requirements. |
| `shuffle_cards`, `random_seed` | True, 53 | Repeatable query order; set seed to None for new ordering. Small batches can stop before every query. |
| `target_passes` | 0 | Zero = full batch; 3 = stop after three accepted under the selected pipeline. Listing/API caps still apply. |
| `max_whitening_cards` | 15 | Hard cap on paid card evaluations: at most 45 calls with the reviewed 2–3-call workflow. This is a card cap, not an exact dollar budget. |
| `max_back_photos` | 8 | Additional photos considered. One sharpest validated back is assessed per listing. |
| `centering_workers` | 2 | Separate processes, avoiding legacy shared-global races. Use 1 for serial debugging, 2–4 on a capable machine. API judgments remain serial. |
| `skip_previous_listings` | False | Skip all recorded IDs under output_root, including earlier rejections/undetermineds. Turn off when retesting fixes. |
| `run_label` | empty | Optional readable folder suffix (letters/numbers/hyphens/underscores). |
| `open_report` | False | Open review.html after a local run. Ignored on GitHub. |

Parallel front processing can finish a few prefetched fronts beyond a reached pass
target; these may remain as audit images without completed result records. They are
never sent to whitening after the target. Cancellation can wait for an already-running
front fit to finish. No speedup factor is promised without benchmarking your machine.

## Local-photo testing

Set `source="local"` and point `local_input_dir` to a folder with one listing per subfolder:

```text
uploads/
  example_01/
    front.jpg
    back.jpg
    back_2.jpg
    metadata.json   (optional)
```

Supported files: JPG/JPEG, PNG, WEBP. Centering requires exactly one image named
`front` with one of those extensions. API-only can use a folder containing only backs.
Optional metadata: `{"card_id":"22","title":"Charizard example","url":"https://www.ebay.com/itm/..."}`.
Provide an approved `card_id` in metadata, or name the folder exactly after an approved
ID (e.g. `eevee_sm233`). The catalog determines the fitter. If you also supply
`border_type`, it must match that ID. Unknown IDs are refused; approved cards outside
the selected family/IDs are skipped. Card-ID selection and per-card limits apply;
eBay age/condition filters do not apply to local photos.

Approval is defined in `approved_borders.json`; changing a search query cannot assign
a card to a different engine. eBay titles must match the approved name, collector
number and GX/non-GX marker. Missing/ambiguous/mismatched identities are skipped before
fitting, counted as `identity_not_approved` in summary.json. This can reduce coverage
for vague titles. It is metadata routing, not visual verification of mislabeled seller
photos. No failed GX result is retried as yellow, and vice versa. API-only mode also
limits sourcing to the chosen approved catalog.

No URLs or keys are needed for local centering. Local whitening still requires an
OpenAI key unless preparing crops only. Use this to retest saved problem images.

## Outputs and resuming

Each run creates a new `outputs/card_runs/TIMESTAMP_LABEL/`; nothing clears earlier batches:

- `review.html`: production or testing view.
- `accepted.csv`, `accepted.json`: accepted cards only, with margins and brief API evidence.
- `results.csv`: all records in testing mode, including individual stage timings.
- `summary.json`: counts, stop reason, API attempts, and estimated cost (not billing).
- Reports and CSVs identify the assigned border type; summary.json includes counts by family.
- `_audit/run.json`: all decisions, effective settings, token usage and errors.
- `_audit/Combined/`, `downloads/`, `details/`, `backs/`: evidence and diagnostics for debugging.

Production mode filters the user-facing report; audit files intentionally retain all
records and images so problems remain reproducible. Results and credentials are gitignored.
Printed centering overlays retain the original engine's "No centering selection gate"
label: the runner applies its selection gate afterward; review.html/CSV show that final decision.

To continue a stopped run on the same computer, set `resume_run` to that run folder.
Pre-border-selector run files cannot be resumed into the new routing version; start
a fresh run. The approved-catalog hash is saved, and a changed catalog requires a new run.
Keep the same pipeline, gate, inputs, reference bank and output mode. You may raise
listing/candidate limits, `max_whitening_cards` or `target_passes` and change worker count.
Completed judgments are never repeated. An API request interrupted before its result
was saved becomes `interrupted_review_required`; its reserved attempt remains counted.
Runs stopped by account/API errors retain evidence and exit with an error. Fix the
account issue before continuing. Change measurement settings in a new run, not by resuming.

Any setting can also be overridden without editing the file:

```powershell
python run_cards.py --set pipeline=centering --set max_listings=10
python run_cards.py --set pipeline=api --set prepare_api_only=true
python run_cards.py --set source=local --set local_input_dir=uploads
python run_cards.py --set resume_run=outputs/card_runs/YOUR_RUN_FOLDER
```

Use JSON values for booleans/lists/null in overrides. The Python SETTINGS block uses
Python values (`True`, `False`, `None`). Do not put secrets in either form.

## Offline verification

```powershell
python -m unittest test_card_runner
python -m unittest discover -p "test_*.py"
python run_cards.py --check
```

These tests use fake network/API responses and local generated fixtures. They do not
start a live eBay batch or spend whitening API credits.

### Back detection tests in GitHub

In **Actions → Run card pipeline → Run workflow**:

- Set **pipeline** to **detection**.
- Set **detection_method** to **shared_baseline** for the existing shared card detector, or **validated_whitening** for the current stricter whitening preparation. These run existing algorithms without changing their thresholds. `shared_baseline` means the current shared implementation, not a frozen historical release.
- Set **max_listings** to your sample size. The usual approved card queries and border selection still apply.
- Leave **replay_run_id** blank to gather new eBay listings. To reuse a saved batch, enter the numeric ID from a previous run's URL (`.../actions/runs/123456...`). Its `card-results-*` artifact must still be available and contain originals and manifests. A review HTML or JPG sheets alone cannot be replayed.

Detection mode always retains every processed listing, ignores centering acceptance and paid-whitening limits, and makes no whitening calls. It does not count a prepared sheet as a passing card. It uses only eBay credentials for fresh listings; saved-batch replay uses downloaded originals without listing searches. Detector source hashes are recorded with each result.

Download the new card-results artifact and open **review.html**. Review the selected original, detected outlines, sheets and other originals. Answer **Did it select a suitable back photo?** and **Does the outline follow the physical card edges?** separately, then add notes. Save reviewed HTML or export JSON; JSON contains detection answer counts and preparation timings. Human accuracy figures require completed reviews.

A replay starts a new output batch, retains listing IDs, and tests the downloaded originals. It does not overwrite the earlier artifact. Older runs missing saved originals stop with an explanation. GitHub artifacts expire after their retention period (currently 14 days).

Detection tests default to validated_whitening. When the existing back outline checks fail,
a perspective contour may qualify if all four physical edges have contrast along their
length, three or more sides have blue ink support, and the rectified photo has the expected
two yellow logo bands and central red ball. At most one shaded side can substitute a strong
dark-inside/light-outside edge for blue ink. This is a heuristic; extensive holder glare may
still require manual review. shared_baseline remains available for historical comparisons.

The whitening detector also matches printed-back landmarks on each original photo, including
photos provisionally tagged as fronts. A match must span the artwork. It predicts where to
look for the border, then measures a color transition on each of the four sides. An outline
from the earlier detector is retained when it agrees with this independent location. A
recognized back with insufficient edge evidence can appear as
`back_photo_found_edges_unconfirmed` in the review. This step runs locally.

The landmark bank in `references/back_landmarks.npz` holds coordinates and SIFT descriptors
for three back designs. The first two source files were the aligned `back_1.jpg` images from the October
20 archive for listings 117476881829 (classic) and 117476782928 (alternate). The gold design uses the archived aligned photo
`257747909612/front_2.jpg`, visually confirmed as a back despite its historical filename. To reproduce
the features, remove the 64-pixel pad to isolate the 1500 by 2100 card body, resize the grayscale
body to 360 by 504 with area interpolation, and apply CLAHE (clip limit 2, 8 by 8 tiles).
Use OpenCV SIFT with 650 features and a mask excluding the outer 27 columns and 32 rows on
each side. Store keypoint locations as float32 and descriptors as uint8. Reference photos
are excluded from the validation samples. Whitening's 16 labeled examples
serve a separate purpose and are still used by the whitening model.
# Pixel candidate tests (separate from API whitening)

In GitHub **Actions → Test pixel whitening candidates → Run workflow**, select:

- **geometry:** triangle, square, circle, or perimeter ring.
- **candidate_strategy:** `edge_uncertainty` adds conservative background rejection before the existing grain/geometry rules; `legacy` restores the previous candidate decisions. Defaults to edge_uncertainty. Use the same replay run and all other settings for comparisons.
- **edge_classifier** is a third, experimental strategy. It classifies all original candidates as likely whitening, likely artifact, or inconclusive using location, local noise/color change and patch shape. The main image shows likely whitening only; the table and audit retain every other mark. Triangle/grain exclusions remain recorded as earlier-stage evidence, not definitive classifier decisions. A strong nearby interior patch can be recovered from triangle rejection. No candidate IDs or human labels are classifier inputs. This was developed on a small labeled set, not validated for general accuracy; the default strategy remains edge_uncertainty.
- **size_pixels:** triangle/square side length, circle diameter, or ring width, normalized to a 500-pixel corner crop. Default triangle side 12 px gives about 10.4 px tip-to-base distance.
- **blue_percentage:** exclusion threshold (default 60%, allowing 40% non-blue).
- **grain_filter:** enabled by default; reject weak small candidates consistent with nearby blue-area noise before geometry. Removed candidates carry G labels. Disable it and replay the same batch for a controlled comparison.
- **exclude_interior_art:** enabled by default for the edge_uncertainty strategy. Large pale rectangles extending through both inward sides of a corner crop are excluded as printed artwork (A labels). The entire candidate must be inside the detected art region, clear of the edge uncertainty zone. Disable this checkbox to compare; legacy ignores it and preserves old decisions.
- **replay_run_id:** paste the numeric ID from a previous Actions run URL to compare settings on saved sheets. Leave blank to fetch a new eBay batch and prepare sheets with the validated detector.
- **max_listings**, **border_type**, and **card_ids:** control the batch. Replay evaluates up to max_listings saved records in original order; border/card search filters only apply to new batches.

Triangle tips are at candidate pixel centroids, with bases pointing toward each named corner: TR 45°, TL 135°, BL 225°, BR 315° in standard mathematical angles. Squares and circles are centered on candidates. Triangle/square/circle percentages include candidate pixels inside the shape. Rings surround the candidate perimeter and also require at least 35% blue in each of eight directions. Shapes cut off by the image remain eligible. A missing sheet is reported, never treated as clean.

Download **card-results-pixel-N**, extract it, and open **candidate_review/review.html** inside the run folder. Red marks are retained candidates, green marks are excluded, and cyan shows the sampled shape (rings are recorded in measurements). The review embeds its photos. Share the HTML and **results.json** with colleagues; **settings.json** records exact parameters and **summary.json** records counts. Original detection results are also preserved for replay.

This workflow does not receive an OpenAI key, does not use the API whitening classifier, and does not run centering. New batches use eBay credentials; replay uses saved sheets without fetching listings. It is an experimental candidate filter, not a card pass/fail test. Lower thresholds remove more spots, including potentially real chips. Counts are not accuracy measurements. The normal **Run card pipeline** workflow is unchanged.

Large spots no longer bypass geometry or the inner-boundary filter. No L labels are generated.

The edge strategy uses three blue silhouettes plus a 4%-of-crop safety margin (20 pixels on a 500-pixel tile). A whole candidate must lie outside every estimate and that margin to receive **B** (background). The margin widens where estimates differ; unreliable fits skip background rejection. Bright neutral regions connected to the uncertain edge also remain uncertain. These are conservative heuristics, not calibrated confidence. Glare, colored backgrounds, and sleeve dust near the edge remain difficult. The original image is never cropped or masked for measurement. Geometry/grain rules can still reject candidates in the uncertainty zone.

With the new strategy, the main review shows retained candidates; expand **Show edge uncertainty and all excluded candidates** to see amber uncertainty and B/G/T labels. All exclusions remain in results.json. Corners with unresolved edges are explicitly noted.

Fallback code is preserved at tag **pixel-fallback-before-edge-uncertainty** (commit b908443). Selecting legacy restores the old decision behavior without reverting the repository. To reproduce the old experiment exactly, also keep the same photos, geometry, size, percentage, and grain setting.
