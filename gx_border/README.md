# Guarded GX border pipeline

Use the common entry point `python run_cards.py --set border_type=gx`, or select
**gx** in GitHub's **Run card pipeline** form. `all` includes both approved families,
but each card still runs exactly one fitter.

This packages the installed `Proj001/gx_guarded_engine` used by
`ebay_gx_line_fit_test_v2.py`, including the October 11 sparse-keyline recovery.
The 38 engine/source-policy files are unchanged. `source_manifest.json` records
their SHA-256 hashes; `.gitattributes` preserves their exact bytes across platforms.
`adapter.py` translates reportable axes to the shared runner format and adds
margin labels to observed-line overlays. It does not change boundary selection.

## Approved designs

The user confirmed the latest active `gx_review_cards.csv` list for this upload:

| Card ID | Design |
|---|---|
| eevee_sm233 | Eevee GX SM233 |
| umbreon_sm36 | Umbreon GX SM36 |
| eevee_sm174 | Eevee GX SM174 |
| greninja_sm197 | Greninja GX SM197 |
| espeon_sm35 | Espeon GX SM35 |
| pikachu_sm232 | Pikachu GX SM232 |
| charizard_sm195 | Charizard GX SM195 |
| primarina_sm39 | Primarina GX SM39 |
| blastoise_sm189 | Blastoise GX SM189 |

Zekrom SM138 remains disabled and excluded. Guzzlord, Gyarados, and the older
GX diversity/experiment lists are not included in the approved catalog.
`data/cards.csv` controls activity and queries within this approved set;
`../approved_borders.json` defines the allowed family and exact identity.

## Measurement rules

GX measures the physical edge to the agreed grey/charcoal keyline using its own
source tracks, common observed spans, frame checks, holder repairs and quality gates.
Unresolved/blocked axes stay blank. Diagnostic candidate values never become margins.
An incomplete or rejected GX fit is never retried in the yellow engine.

The shared centering gate (default 53/47) is applied only after both GX axes are
reportable. In combined mode, only those centering passes continue to Luna.
GX does **not** inherit yellow's color thresholds, perspective cutoff or ±1-point
angular allowance. The existing GX reportability checks remain responsible for
withholding uncertain measurements. Any GX hard failure fails selection.

Design approval does not establish universal measurement accuracy: the imported
GX implementation remains the reviewed, guarded pipeline, with uncalibrated
unseen-listing accuracy. Exact-image review vetoes in `reviewed_rejections.json`
are preserved as safeguards, not represented as a general image-quality detector.

## Routing limits

The runner checks active approved IDs, exact card name/collector number and the GX
marker in listing titles before downloading/analyzing a front. Ambiguous titles,
mixed-card lots, unapproved IDs and conflicting family tags are excluded or refused.
This prevents software routing crossover. It does not visually verify that a seller
has photographed the card named in the title. Local photos likewise rely on their
explicit approved card ID. Misidentified photos need a separate identity/OCR check.

## Offline tests

```powershell
python -m unittest test_border_routing test_gx_final_frame test_gx_candidate_identity
```

The frame/candidate tests were ported from the GX development regression suite;
only their import paths changed. Routing tests also verify strict fitter separation,
no fallback, no yellow allowance, withholding and release-file hashes.
