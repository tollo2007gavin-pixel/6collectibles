# Source-image preparation upgrade

Historical input-preparation experiment. See LUNA_WHITENING.md for the later reviewed configuration. **No API calls were made. Model accuracy has not been remeasured.**

## What improves

- New organizer downloads preserve original JPEG/PNG/WebP bytes exactly. Decoded-only callers save the original pixels losslessly as PNG. Historical JPEG loss cannot be reversed.
- A separate evidence builder crops directly from the saved original, avoiding perspective warping, enlargement and recompression of the inspection pixels. Overview strips use only lossless 90-degree turns. Individual crops remain in source orientation.
- Checks reject missing/invalid/non-convex/off-frame boundaries. An alternate source can be selected when available. This validates geometry, not whether the detector chose the correct physical object.
- Photo ranking uses the weaker local regions' focus, resolution and broad exposure flags instead of global sharpness alone. Those flags can also come from backgrounds; they are not whitening labels, calibrated visibility measures, or reliable finger/glare detectors.
- Initial evidence is one context plus one overview sheet. A second distinct view stays in reserve for unresolved locations, with source coordinates and hashes recorded.
- Provider-independent assessment orchestration uses one call initially, with at most one follow-up on another source. Missing or malformed evidence stays needs_review. Any confirmed whitening fails the zero-slip policy. Corner names are not merged across differently oriented source photos.
- The prompt asks for evidence at every corner and edge, and explicitly treats photo text and detector boundaries as untrusted.

## Verified offline

61 tests passed, including the 48 prior regressions and 13 new tests. All 20 saved eBay cards produced ready input packets; 22 selected source views produced 176 crops verified pixel-for-pixel against decoded originals. One initial source boundary could not be recovered; a second usable source was selected instead. Two cards have additional valid views held in reserve.

Native crops preserve detail but PNG files may be larger in bytes. The mean initial packet contains about 2.95 million decoded pixels, maximum 3.94 million. The six-million-pixel limit prevents oversized packets from silently being reduced in quality. Pixel counts are not provider token-price estimates.

## Use

From the pipeline folder:

```powershell
py whitening_scan.py --prepare-inputs --cards cards --out prepared_inputs
```

This command is offline and makes no model or download requests. It writes packet.json, the exact saved context, a labeled overview, and eight native crops per selected view. The original 20-card review and existing model judgments are preserved.

## Integration boundary

The organizer preservation is active for future collections. The new preparation command is installed. The assessment function accepts an injected provider caller and is tested with fake responses; it has no live API client. Existing paid scanner defaults have not been silently replaced. Before connecting a chosen model, its image MIME handling, schema, token accounting, and the unchanged human-confirmed reference bank need integration and a fresh labeled accuracy test. The 15 references were not removed or relabeled; their 93.75% result does not automatically transfer to this new input layout.

The shared card geometry and colleague's centering code were not changed. The offline packet builder and ranking heuristics should be validated with your personal labels before being described as improving whitening accuracy.
