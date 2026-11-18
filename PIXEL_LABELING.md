# Mark cards and compare improvements

This system turns your review marks into reusable data. It does not infer labels from notes, filenames or an earlier model's answer. It makes no OpenAI calls. GitHub Actions runtime remains subject to your GitHub plan.

## 1. Collect and mark a batch

Run **Actions → Test pixel whitening candidates** with **candidate_strategy = edge_classifier**. Open `candidate_review/review.html` from the downloaded artifact. Other strategies still have marking tools, but automatic comparison requires the classifier's saved measurements.

On a fresh batch, click **Reserve about 30% of unreviewed card groups** before reviewing. This is a stable group-based assignment, so small batches may not divide exactly 70/30. You can choose Development or Reserved validation manually before making marks. Previously reviewed examples belong in development. The grouping and split lock when reviewing starts, including after undoing your marks.

Open **Mark whitening, glare and missed chips** on a card:

1. Choose its corner and zoom level.
2. Choose **Whitening**, **Not whitening**, or **Uncertain**.
3. Click a blue candidate box to label it, or drag a tight rectangle around a missed chip. Draw separate rectangles for separate chips. Hide candidate boxes when they obscure the photo.
4. Use **Undo last edit** or select a mark and **Delete selected mark** to correct mistakes. Changing the selected label and clicking a mark relabels it.
5. Check **I reviewed this whole corner and marked every whitening chip** only after a complete inspection. Otherwise, unmarked areas remain unreviewed. Uncertain marked regions are excluded from scoring.
6. Keep notes for context. Notes never become labels automatically.

The **Physical card group** identifies the actual card, not the photo. Different images or listings of the same physical card must use the same group. Identical source photos and matching listing IDs are also checked automatically. The software cannot identify undisclosed relistings or all visually similar duplicate photographs.

Click **Save marked review with photos** before closing. It produces an editable HTML file with the original photos, labels and notes. **Export labeled dataset JSON** creates the smaller file used for comparison. **Import labels** resumes a compatible export against the same photos. The page does not automatically upload your work. Keep files from separate batches under distinct filenames.

After validation has guided another change, use **Move reserved cards to development** to retire them. They cannot be reserved again through the page. A new independent check requires new reserved cards.

## 2. Run automatic comparisons in GitHub

1. Open the repository's `review_labels` folder.
2. Choose **Add file → Upload files** and upload your exported labeled JSON. For a revised review, replace its earlier export instead of keeping conflicting copies. Upload a complete file, not text pasted into workflow fields.
3. Open **Actions → Learn from pixel whitening reviews → Run workflow**.
4. Leave **dataset_path = review_labels** and choose an action:

| Action | What it does |
| --- | --- |
| baseline | Scores the current settings on development labels only. |
| tune | Compares 23 bounded settings: the current core thresholds, then each core threshold at 80% and 120%. Selects an improvement in F1 only if development recall does not fall. Validation answers are not scored. |
| validate | Scores reserved cards using a configuration frozen by a previous development run. No settings search takes place. |

Download the **pixel-learning-report** artifact. Read `REPORT.md`; `report.json` contains counts and per-card details. Baseline and tuning also output `frozen-config.json`. Results are proposals: the workflow does not modify the active detector or push code.

Tuning needs at least three reviewed physical-card groups, three positive regions, and either three explicit negative regions or a fully reviewed corner. This is only a minimum to run, not proof that the dataset is large enough for reliable generalization. With too few labels it reports the limitation and freezes the baseline without claiming improvement.

## 3. Validate without using the answers to tune

Finish development first. Upload its `frozen-config.json` into `review_labels`, then choose **validate** and enter `review_labels/frozen-config.json` as the configuration path. The program checks code/configuration compatibility and rejects training/validation overlap through image hashes, normalized listing IDs and physical-card groups, including transitive links.

Validation outputs `validation-exposure.json`. Save it back into `review_labels` alongside your exports. Later validation runs read these records and refuse to treat the same cards as fresh validation. Preserve records from different validations under distinct filenames. If you discard a record or evaluate outside this system, the tool cannot know the answers were previously used. Re-running a report on the same reserved examples is not a new independent trial.

Once you inspect validation failures and use them to change the detector, move those cards to development and reserve new cards. Changing a validation card's labels must never be used as a way to select settings.

## Reading the results

- **Recall:** proportion of your marked whitening regions detected.
- **Precision:** proportion of scored positive predictions that match marked whitening rather than reviewed negatives. It is not reported without negative coverage.
- **Missed regions:** marked chips without a matching positive prediction, with separate counts for no matching extracted candidate, inconclusive candidates and artifact classifications.
- **False-positive marks:** positive predictions in explicit negative regions or fully reviewed corners.
- **Ignored marks:** predictions in unreviewed or uncertain regions; they are not quietly treated as correct or incorrect.

Matching uses corner-local boxes: intersection-over-union of at least 0.1, or at least 50% overlap of the smaller box with no more than a 25-fold area difference. A prediction can match only one chip. Multiple fragments on the same labeled chip do not inflate the true-positive count. Avoid huge or overlapping chip rectangles; those make localization measurements less useful. Drawn boxes are measurements on the existing enlarged sheets, not physical defect dimensions.

These metrics apply only to the labeled regions. Marking only spots already found by the detector can produce deceptively high recall: draw its missed chips too. A mix of clean corners, tiny chips, sleeves, glare, textured backgrounds, and different image qualities is necessary for useful comparisons.

## Limits and reproducibility

This first version tunes core classifier thresholds using saved measurements. It does not train a neural network or modify extraction, outline detection, artwork rejection or the wider-patch rules. If no candidate was extracted at a chip, classifier tuning alone may not recover it. The annotations still record that failure for future detector development.

Results include image identities, candidate measurements and code hashes. Incompatible feature caches are rejected with a request to replay the original sheets; they are not silently mixed. Frozen configurations include matching policy, core and fixed patch thresholds, code fingerprints and development-data identities. No centering or production pipeline logic is altered by the marking or learning workflow.

For local use:

```text
python pixel_training.py baseline --dataset review_labels --out outputs/baseline
python pixel_training.py tune --dataset review_labels --out outputs/tuning
python pixel_training.py validate --dataset review_labels --config review_labels/frozen-config.json --out outputs/validation
```

Output folders must be new. JSON uploads contain listing identities, measurements, grouping and your judgments. Photos stay in the separately saved HTML review.
