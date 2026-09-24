# Setting up API keys and running the whitening test

## 1. Get the keys
| Service | Where | What you get |
|---|---|---|
| eBay Browse API | developer.ebay.com → Application Keysets → **Production** | App ID (Client ID) + Cert ID (Client Secret) |
| OpenAI | platform.openai.com → API keys → *Create new secret key* | `sk-...` key (shown once, copy it immediately) |
| Gemini (optional) | aistudio.google.com → *Get API key* | key for the free tier / paid tier |
| Anthropic (optional) | console.anthropic.com → API keys | `sk-ant-...` key |

Before using any key, set a **monthly spend limit** in that provider's billing page
(e.g. $10 while testing). A leaked key can then only cost you that much.

## 2. Store them locally (your laptop)
```bash
cp .env.example .env      # then open .env and paste the values after the = signs
```
The scripts load `.env` automatically. It is in `.gitignore`, so it never reaches GitHub.
Real environment variables override `.env` if both are set.

## 3. Store them for GitHub Actions
Repo → Settings → Secrets and variables → Actions → *New repository secret*.
Add each key under the same name as in `.env` (`OPENAI_API_KEY`, etc.).
The workflow reads them as `${{ secrets.OPENAI_API_KEY }}`; they're masked in logs.

## 4. Key hygiene
- Never paste a key into code, a commit, a screenshot, Discord or an AI chat.
- If a key has been shared anywhere like that, **revoke it and create a new one**. It takes a minute.
- One key per person/machine makes it easy to revoke just one.

## 5. Consistency test (run this first)
```bash
pip install -r requirements.txt
python ebay_fetch.py --config searches.yaml                       # pull listings (+ centering)
python organize_listings.py --from-results out/results.jsonl --out cards
python whitening_scan.py --dry-run --photos all --sides both      # $0: check cards/*/sheet_*.jpg by eye
python whitening_scan.py --provider openai --photos all --sides both \
       --min-photos 2 --repeats 3 --max-cards 10                  # ~30 calls, a few cents at most
```
Read `cards/consistency_report.json`:
- `overall_full_agreement_pct` — share of cards scored identically on all 3 runs. Aim for 80%+.
- `location_disagreement_pct` — which corner/edge the model flip-flops on.
- `least_consistent_cards` — open those listings and look at them yourself.

If agreement is low: tighten the severity definitions in the prompt box, try
`--photos best` (sometimes extra blurry photos add noise), or compare `--provider gemini`.
Re-run a changed prompt on the same cards with `--rescan`.
