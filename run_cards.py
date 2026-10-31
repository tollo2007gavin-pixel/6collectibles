"""Edit SETTINGS below, then run this file in VS Code: python run_cards.py.

GitHub: Actions > Run card pipeline > Run workflow. See RUN_CARDS.md.
Keys belong in .env or GitHub Actions secrets, never in this file.
"""
from card_runner import Settings, main

# ===================== YOUR RUN SETTINGS =====================
SETTINGS = Settings(
    pipeline="both",                 # "both", "centering", or "api"
    output_mode="testing",           # "production": accepted-only summary; "testing": every result + images
    source="ebay",                   # "ebay" or "local" (reuse photos without fetching)
    check_only=False,                # True validates setup only: no fetching or paid calls

    max_listings=45,                  # Total eligible listings analyzed, NOT number of passing cards
    max_listing_age_hours=None,       # None = any age; 24 = listed within the past day
    max_listings_per_card=3,          # Per active cards.csv search; one front per listing
    max_candidates_per_card=100,     # Search/detail budget, including unsuitable listings
    card_ids=[],                     # [] = all active CSV rows; e.g. ["22", "23"]
    cards_csv="yellow_border/data/cards.csv",
    shuffle_cards=True,
    random_seed=53,                   # None = fresh order each run
    skip_previous_listings=False,    # True skips IDs in earlier runs under output_root

    centering_max_side_pct=53.0,      # 53 means 47/53 or better on BOTH axes
    allow_angular_variance=True,     # Existing +/−1 point overlap for qualifying angled photos
    target_passes=0,                 # 0 = process full batch; e.g. 3 = stop after three accepted
    max_whitening_cards=15,          # Hard cap on paid card evaluations (up to 3 API calls each)
    prepare_api_only=False,          # True builds crops without paid calls; never counts as a pass
    max_back_photos=8,               # Additional listing photos examined for a usable back
    references_json="luna_references.json",

    centering_workers=2,             # Separate CPU processes; 1 for serial debugging
    output_root="outputs/card_runs", # Every run gets a NEW folder; previous results are preserved
    run_label="",                    # Optional folder label, e.g. "charizard_lighting"
    resume_run=None,                 # Path to a previous run folder; same measurement settings required
    local_input_dir="uploads",      # One folder per listing: front.jpg, back.jpg, back_2.jpg...
    env_file=".env",
    open_report=False,               # Open review.html when a local run finishes
)
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main(SETTINGS))
