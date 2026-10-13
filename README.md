# Six Collectibles – listing pipeline

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
