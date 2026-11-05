# Six Collectibles – listing pipeline

## Start here: one configurable runner

Use **[run_cards.py](run_cards.py)** for centering only, reviewed Luna whitening only,
or centering followed by whitening. Edit the settings at the top, or use
**Actions → Run card pipeline → Run workflow** on GitHub.
Choose accepted-only production summaries or testing reports with images and reasons.
Select **yellow**, **gx**, or **all** approved border designs. Each card uses only
its assigned engine; the [GX package](gx_border/README.md) preserves its own quality checks.
See **[RUN_CARDS.md](RUN_CARDS.md)** for setup, inputs, output files, and local-photo testing.
Install `requirements_runner.txt` for this unified runner. Secrets stay in `.env` or Actions secrets.

The commands below remain available as individual/legacy tools.

The standalone [yellow-border centering pipeline](yellow_border/README.md)
is available in `yellow_border/`, including its eBay batch runner, configuration,
and offline tests. It has separate dependencies and outputs from the workflow below.

```
ebay_fetch.py        -> pulls new listings, runs centering, writes out/results.jsonl
organize_listings.py -> one folder per card, every photo flattened + tagged front/back
whitening_scan.py    -> OpenCV score + one vision-API call per card on a zoomed edge/corner sheet
```

## Run order
```bash
python ebay_fetch.py --config searches.yaml
python organize_listings.py --from-results out/results.jsonl --out cards
python whitening_scan.py --cards cards --dry-run          # free: builds sheets + OpenCV scores
python whitening_scan.py --cards cards --provider gemini  # or openai / anthropic
```
Manual uploads: put each card's photos in its own sub-folder, then
`python organize_listings.py --from-dir uploads --out cards`.

## Keys (env vars)
EBAY_CLIENT_ID, EBAY_CLIENT_SECRET, and one of OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY.

## Tuning
- Prompt: the boxed "EDIT THE PROMPT HERE" section at the top of whitening_scan.py.
- Prices: PRICES in providers.py (checked 2026-09-23; re-check before budgeting).
- `--gate`: skips the API when the OpenCV score is clearly clean/damaged. Calibrate
  `--gate-clean` / `--gate-heavy` from dry-run scores on cards you've judged by eye first.


## Reviewed Luna whitening

See [LUNA_WHITENING.md](LUNA_WHITENING.md) for the optimized Luna configuration, automatic crop checks, local calibration setup, offline tests, and the latest human-reviewed results.
# Reviewing and sharing test results

Keep the existing **Run card pipeline** dropdown settings and select **testing**
to include every decision. Download and extract the resulting `card-results`
artifact, then open `review.html` in a browser. Completed runs embed available
photos, so the HTML also works independently of the extracted image folders.
Production reports continue to show accepted cards only.

The review shows one card at a time with its pipeline decision, evidence,
original photos, crop sheets and any saved centering/outline views. Use
Previous/Next, the card selector or Next unreviewed to navigate. Mark the
decision **Correct**, **Incorrect** or **Unsure**, optionally supply your own
decision, and add notes. Human answers are never prefilled from model results.

- **Save reviewed HTML** downloads a portable copy with photos and feedback.
  Send that file to colleagues; opening it restores the embedded answers.
- **Export review JSON** downloads a compact, structured record containing
  pipeline decisions, human feedback, notes and image hashes. Share it with
  chat models or use **Import review JSON** to resume a review.
- Browser autosave is convenient but browser-specific. Download backups
  regularly. The page does not upload feedback or photos anywhere.

Live checkpoints use relative photos to avoid rebuilding a large embedded page
after every card. The final report embeds images once. To upgrade an older
extracted result without new API calls:

```sh
python card_review.py path/to/extracted/run-folder
```

This resolves archived GitHub-runner paths against the local `_audit` folder.
Missing photos are shown as unavailable rather than fetched from the internet.

### Correct missing or misaligned sheets

Open **Fix missing or misaligned corner sheets** in the portable review. Select
an original back photo, choose **Mark corners**, then click **top left, top
right, bottom right, bottom left** on the card itself. Use the intersections of
the straight edges at rounded corners, not the sleeve. All four corners must
be visible. Choose **Save four corners**, then **Save reviewed HTML**.
Corrections and notes also survive JSON export/import. Nothing is uploaded.
Changing notes does not erase saved corners. Unsaved points are discarded when
you change cards; save the four points first.

The saved correction identifies the exact original file, SHA-256, dimensions,
and normalized corner coordinates. To generate sheets locally, run:

```sh
python review_corners.py "reviewed-batch.html" --out "outputs/corrected-batch"
```

The output folder must be new. This prepares only cards with saved corrections,
creates an original, outline, four-corner/edge sheet and corner close-ups, and
opens no network connections. Open its `review.html` to check the new crops.
Earlier decisions and human notes remain in `_audit/run.json`; the new review
says **not run**, because new crops have not yet been assessed. It does not
silently reuse the old clean/not-clean prediction.

To prepare and reassess the corrected cards with the existing reference bank,
use a different new output folder and explicitly enable paid calls:

```sh
python review_corners.py "reviewed-batch.html" --out "outputs/corrected-reassessed" --run-whitening --references "luna_references.json" --env-file ".env"
```

This uses the same Luna classifier and reference selection as the runner,
skips centering and listing searches, and checkpoints each reassessment.
Without `--run-whitening`, no whitening calls occur. The downloaded review
cannot launch Python or API calls itself; the command runs on your computer.
Do not commit downloaded listing photos, completed reviews, or credentials.

Automatic detector version `supported-outer-lines-4` first preserves successful
version-3 detections. When those fail, it adds line-segment candidates and a
blue-tinted-background recovery pass. That pass requires stronger continuous
edge support and a large saturation drop toward the background; it does not
simply accept every blue exterior. Failed localization still blocks whitening.
This improves coverage for some photos; it is not a guarantee that every crop
is correct or a new measurement of whitening accuracy.
