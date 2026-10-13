# Yellow-border centering pipeline

Standalone production runner copied from Proj001. This folder does not replace
the repository's listing/whitening scripts. It makes eBay requests but no
OpenAI, Gemini, or Anthropic calls.

## Setup in VS Code (Windows)

Use Python 3.11 or newer. From the repository folder:

```powershell
cd yellow_border
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_line_fit.txt
Copy-Item .env.example .env
```

Fill `EBAY_CLIENT_ID` and `EBAY_CLIENT_SECRET` in the local `.env` with your
eBay Production application keys. Alternatively, supply them as environment
variables. `.env` is ignored by Git. No actual credentials are included.

In VS Code, select `yellow_border/.venv/Scripts/python.exe` as the interpreter,
open `ebay_line_fit_test.py`, and choose **Run Python File**, or run:

```powershell
.\.venv\Scripts\python.exe ebay_line_fit_test.py
```

For macOS/Linux, use `.venv/bin/python` and `cp .env.example .env` instead.

## Configuration and behavior

- `data/cards.csv` controls active card searches. The included configuration
  contains the current yellow-border GG/TG searches; inactive rows are not run.
- The runner's top-level settings control batch size, listings per card,
  listing age, and shuffling. Defaults: 45 fronts, 3 listings per card.
- Only seller-listed ungraded / Near mint or better listings qualify.
- Only the main listing image is downloaded, assumed to be the card front.
- Adaptive and forced-curve fits are combined. Supported forced-curve paths
  are drawn preferentially; qualified method percentages are averaged.
- The rebuilt perspective frame is used where supported. The perspective
  limit is 0.20, with a proposed +/-1 percentage-point angular allowance above
  0.15. This allowance is not a calibrated statistical confidence interval.
- Missing-exterior-background failures and other measurement checks remain
  active. No centered-card selection gate (45/55, 47.5/52.5, etc.) is enabled.
- Background contamination, interrupted-edge, lighting variation, and
  wider-boundary-pair recoveries are included.

Outputs go to `yellow_border/centering_output/ebay_line_fit/`: combined overlays,
original working images, detailed JSON, a CSV, and `review.html`.
Each run replaces its managed batch outputs. Separate review folders are preserved.
Results, image datasets, virtual environments, and secrets are not included here.

## Offline verification

From `yellow_border`, run the tests without downloading listings or needing keys:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s . -p "test_*.py"
```

See [the detailed pipeline guide](EBAY_LINE_FIT_README.md) for measurement
definitions, failure recovery, and output fields. References there to earlier
review folders describe local development evidence, not files shipped here.
