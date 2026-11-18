# Upload your exported labels here

In a pixel inspection review, mark chips and false alarms, then choose **Export labeled dataset**. Upload those JSON files to this folder using GitHub's **Add file → Upload files**. Keep files from different batches under different names.

Then open **Actions → Learn from pixel whitening reviews → Run workflow**. Start with **baseline**, then use **tune** for automatic development-only settings comparisons. No OpenAI or eBay credentials are used. These actions do not update the active detector.

Keep all photos of one physical card under the same group ID. Known examples already used to adjust the code belong in development. Reserve fresh cards before labeling them; validation answers must not be used to choose settings.

For **validate**, upload the frozen configuration from a completed development report and supply its repository path. Preserve the validation exposure record with your dataset after looking at the results. These cards have then been used for validation; use fresh reserved cards for the next independent check after further tuning.

Do not upload API keys or unrelated files. Label exports contain listing IDs, measurements and your judgments, but not the listing photographs. The saved review HTML contains the photos and is kept separately.
