# Whitening triage v2

`py whitening_scan.py --triage-v2 --cards cards` runs experimental whitening triage v2.
The default command remains the existing zero-slip scanner: the 20-card evaluation showed
that v2 misses too much labeled whitening to replace it. V2 is implemented and available for
further evaluation, not certified as an accurate detector. The centering code is not invoked.
Existing `whitening.json` results are preserved when using v2.

## Outcomes

- **clean:** all requested locations are inspectable and judged clean. Zero confirmed whitening is allowed.
- **whitening_detected:** both independent review calls confirm whitening at the same location with sufficient reported confidence. Other locations may remain uncertain.
- **needs_review:** obscured edges, low confidence, disagreements, missing photos, failed calls, or unresolved evidence. These cards remain in a separate queue; they are not labeled damaged.

The primary model runs three times. If all locations are unanimously clean, the card passes.
Otherwise the disputed locations, earlier evidence, and full images go to a second model twice.
That review can identify a reflection or normal card edge and clear the card, confirm real wear,
or leave it uncertain. Initial suspicions are retained in the result even when overturned.
Review calls cannot rescue failed or incomplete primary calls. No automatic retry is used to hunt
for a preferred answer. Repeated judgments are not independent physical measurements.

## Photos and model settings

Defaults: GPT-6 Luna primary, GPT-6 Sol review; 3 primary calls and 2 review calls when needed;
confidence threshold 0.6; back only; up to two distinct usable back photos, each supplied as
uncropped context, edge sheet, and tip sheet. Confidence is self-reported, not calibrated.

The organizer now saves `original_<side>_<n>.jpg` and records `original_file` in its manifest.
For older cards the scanner retrieves uncropped photos only from HTTPS `i.ebayimg.com` URLs,
with a timeout and 15 MB size limit, and caches them under its output folder. No redirects are followed.
If retrieval fails, it uses the normalized card as context and records that limitation.
Two crops from one photo are explicitly described as one view, not corroborating photos.
Existing side/outline detection remains unchanged; failed detection yields needs_review.

## Commands

```text
py whitening_scan.py --triage-v2 --cards cards --only selected_ids.txt --out whitening_trial
py whitening_scan.py --triage-v2 --cards cards --only selected_ids.txt --out whitening_trial --rescan
py whitening_scan.py --triage-v2 --cards cards --dry-run --no-download-originals
py whitening_scan.py --triage-v2 --cards cards --sides both --max-photos 2
```

`--only` accepts a card-ID list or CSV with a card_id column. Use `--max-cards` to bound work.
`--photos best` uses one photo per requested side; default `--photos all` uses up to `--max-photos`
per side. `--review-model` and its alias `--confirm-model` choose the independent-review model.
`--review-repeats` must be at least two. Only `--allowed-slips 0` is accepted.
`--all-repeats` is retained for compatibility; all repeats always run in v2.

The v2 workflow supports OpenAI. Legacy `--gate`, `--examples`, and `--min-photos` flags are
not part of this workflow and will be rejected instead of silently changing behavior. Reference
examples are deliberately excluded from the reported evaluation. Original helper functions and
`legacy_main()` are retained for compatibility and regression testing. The CLI requires the explicit
`--triage-v2` flag to use the new workflow; `py whitening_workflow.py` is also a direct v2 entry point.

## Outputs and caching

Default output directory: `cards/whitening_v2/`. Per-card JSON includes all primary and review
evidence, source photo identifiers, token counts, estimated API cost, decision, and reason.
`shortlist_v2.csv`, `needs_review.csv`, and `whitening_detected.csv` describe the current invocation.
The old `shortlist.csv` is not rewritten; use the new queues when using v2.

Cache fingerprints include the implementation/prompt, model and policy settings, and transmitted
image bytes. Changed inputs force reassessment. Failed calls are retryable. Estimates are not
billing statements; an unknown model price or failed call produces an unknown cost rather than zero.

Run `py test_whitening_workflow.py` for v2 tests and `py test_regressions.py` for the retained legacy
and ingestion regression tests. The 20-card experiment uses `run_twenty.py` in the audit bundle;
labels are only used by its evaluator, never supplied to either model.

Use `py score_whitening_v2.py --labels labels.csv --results cards/whitening_v2` to evaluate v2.
It counts needs_review separately, includes missing labeled cards, and never counts an abstention
as successful whitening detection. The older `testset.py score` still evaluates legacy results.
