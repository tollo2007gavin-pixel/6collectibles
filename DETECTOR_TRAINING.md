# Train the whitening detector on Windows

Your green boxes tell Faster R-CNN where a chip is. The model learns from the **original corner photos** in your saved reviews. Training runs on your computer. The first launch needs internet access to install the training software and obtain the model weights.

Local training uses your computer's time and electricity. TorchVision distributes its software under the [BSD license](https://github.com/pytorch/vision/blob/main/LICENSE). For commercial deployment, check the [pretrained weight terms](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights), which can depend on the original training dataset.

## First run

1. Open your updated project folder in File Explorer and double-click **Train whitening detector.cmd**.
2. Put your saved photo reviews in **detector_workspace → reviews**. Earlier files named **reviewed-pixel-markings.html** work too. Keep one current copy per batch. If you fix boxes later, replace the old copy. The small GitHub labels file contains no photos.
3. Choose **1** to train a new model. Leave the window open while it runs. Its checkpoint (saved model) and report go under **detector_workspace → runs**.

## Add your next batch

1. Run **Test pixel whitening candidates** as usual. Open the downloaded review and inspect all four corners of each included card. Click a candidate for a real chip or draw one green box per missed chip. If a whole card is clean, write **No whitening** in its notes. Mark unclear spots **Uncertain**; the trainer skips those corners. Leave completely unreviewed cards blank.
2. Click **Save for detector training (.html)**, then move that file from Downloads to **detector_workspace → reviews**. On older pages, use **Save local photo backup** or **Save marked review with photos**. Keep the files from earlier batches there too.
3. Double-click the launcher again. Choose **2** to continue your latest checkpoint with the accumulated reviews. Choose **1** when you want a fresh model starting from the initial weights.

## Test progress

Use **3** to test the latest checkpoint on fresh reserved cards. Add a newly reviewed batch when you need fresh cards. Keep photos of the same physical card in the same group, and mark every card the same way regardless of its assigned split.

For a fresh detector test, save the marked photo review locally **before uploading that batch for pixel-rule validation**. The trainer honors both experiments’ records of previously scored cards. Option 3 uses frozen confidence settings and compares the latest detector with its previous checkpoint when available and with the saved pixel predictions. After a successful test, the launcher records those cards as used. You can reopen the same report whenever you like.

Options 1 and 2 each run three additional passes over the training corners by default. Every completed run gets a separate folder with its settings, metrics and predicted coordinates. The development report helps you compare training runs. Option 3 measures performance on fresh groups that did not guide training or confidence selection. Keep the whole `detector_workspace` folder when moving to another computer, including its checkpoints, reviews and test history.

Look at **chips found**, **chips missed**, and **false alarms** together. Finding 9 of 10 marked chips is 90% recall. Having 9 of 10 flags match real chips is 90% precision. Keep the report with the checkpoint to identify exactly which model produced that result. Once test results influence your next change, use different, fresh cards for the next independent test.

Open **results.html** in a run folder to see the original corner photos. Toggle the green boxes you drew and the orange boxes the model predicted to inspect misses and false alarms.

Choose **4** to repeat the last test with its original checkpoint, confidence cutoff, and exact saved dataset. This creates a separate replay report. Use **2** for additional training and **3** for new evidence of accuracy.
